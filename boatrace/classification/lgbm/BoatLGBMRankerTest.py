from _datetime import datetime
import json
from logging import getLogger, config
import sys

from boatrace.common.BoatEnum import DelimiterType
from boatrace.util.PropertyUtil import PropertyUtil
import pandas as pd
import math
from sklearn.metrics._ranking import ndcg_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

import lightgbm as lgb
import os

#
# model生成するクラス
#
class BoatLGBMRankerTest:
    def __init__(self):
        self._logger = getLogger('server')
        
    # @param param_list_str ex)  boosting_type=gbdt,learning_rate=0.1...
    # @param csv_filepath ex)  D:/Dev/experiment/expr10/arff/9997_rank1.csv
    # @param model_filepath ex)  D:/Dev/experiment/expr10/evaluation/model_release/9997/1/nopattern/9997_nopattern_yyyymmdd_rank1.model
    # @param feature_name_list_str ex)  nw1,nw2,nw3,nw4,nw5,nw6,class
    # @param feature_type_list_str ex)  float,float,float,float,float,float,category
    def execute(self, param_list_str, csv_filepath, model_filepath, feature_name_list_str, feature_type_list_str):
        
        #feature명 리스트 취득
        feature_name_list = feature_name_list_str.split(DelimiterType.DELIM_COMMA.value)
        
        #feature type 리스트 취득
        feature_type_list = feature_type_list_str.split(DelimiterType.DELIM_COMMA.value)
        
        #csv data type 設定
        #data_type = {'nw1': 'float', 'nw2': 'float', 'nw3': 'float', 'nw4': 'float', 'nw5': 'float', 'nw6': 'float', 'class': 'category'}
        feature_num = len(feature_name_list)
        data_type_dict = {}
        cat_features = []
        for i in range(feature_num):
            data_type_dict[feature_name_list[i]] = feature_type_list[i]
            if feature_type_list[i] == 'category'  and feature_name_list[i] != 'class':
                cat_features.append(feature_name_list[i])
        
        # csv data 取得
        df = pd.read_csv(csv_filepath, names=feature_name_list, dtype=data_type_dict, engine='python')
        df = df.sort_values([ 'raceid', 'waku'])
        
        keys = df['raceid'].unique()
        date_split_val =  keys[math.trunc( len(keys) * 0.6)]
        date_split_test =  keys[math.trunc( len(keys) * 0.8)]
        
        # date_split_val = 202006020000
        # date_split_test = 202106012400
        
        #test_size = 0.2
                
        df_train = df[df['raceid'] <= date_split_val]
        #df_val = df[ (df['raceid'] > date_split_val)]
        df_val = df[ (df['raceid'] > date_split_val) & (df['raceid'] < date_split_test) ]
        df_test = df[df['raceid'] >= date_split_test]
        self._logger.info(f"df_train length={len(df_train)}")
        self._logger.info(f"df_val length={len(df_val)}")
        self._logger.info(f"df_test length={len(df_test)}")
        
        X_train = df_train.drop(['class'], axis=1)
        X_val = df_val.drop(['class'], axis=1)
        X_test = df_test.drop(['class'], axis=1)
        
        y_train = df_train['class']
        y_val = df_val['class']
        y_test = df_test['class']
        
        # class 값 확인 (1=1위인지, 6=1위인지)
        self._logger.info("[Class 값 분포 확인]")
        self._logger.info(f"  y_train unique values: {sorted(y_train.unique())}")
        self._logger.info(f"  y_train 첫 레이스 샘플: {y_train.head(6).tolist()}")
        
        # LGBMRanker는 높은 label = 상위 순위로 학습함
        # class가 순위(1=1위, 6=6위)라면 반전 필요: 7 - class
        # class가 이미 점수(1=6위, 6=1위)라면 그대로 사용
        if y_train.min() == 1 and y_train.max() == 6:
            self._logger.info("  → class가 순위(1=1위)로 판단됨. Label 반전 적용 (7 - class)")
            y_train = 7 - y_train
            y_val = 7 - y_val
            y_test = 7 - y_test
        
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
    
        #モデル파라미터 설정
        model_param_list = param_list_str.split(DelimiterType.DELIM_COMMA.value)
        model_param_dict = {}
        for param in model_param_list:
            key, value = param.split(DelimiterType.DELIM_EQUAL.value)
            # 숫자 타입 변환
            if value.replace('.', '').replace('-', '').isdigit():
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            model_param_dict[key] = value

        # class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
        # class_weights = dict(zip(np.unique(y_train), class_weights))
        # print(class_weights)
        # model_param_dict['class_weight'] = class_weights

        # 모델 생성
        self._logger.info(f"Model parameters: {model_param_dict}")
        model = lgb.LGBMRanker(**model_param_dict)
        
        train_group = X_train['raceid'].value_counts()
        train_group = train_group.sort_index()
        
        eval_group = X_val['raceid'].value_counts()
        eval_group = eval_group.sort_index()
        
        # model.fit with early stopping
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50)
        ]
        model.fit(
            X_train, y_train, 
            group=train_group,  
            eval_set=[(X_val, y_val)],  
            eval_group=[list(eval_group)],
            callbacks=callbacks
        )
        self._logger.info(f"Best iteration: {model.best_iteration_}")
        self._logger.info(f"Model params: {model.get_params()}")

        # ========================================
        # 모델 성능 평가
        # ========================================
        self._print_model_performance(model, X_train, y_train, X_val, y_val, X_test, y_test, df_test)
        
        # ========================================
        # Feature Importance 출력
        # ========================================
        self._print_feature_importance(model, df)
        
        # ========================================
        # 모델 저장
        # ========================================
        self._save_model(model, model_filepath)
    
    def _print_model_performance(self, model, X_train, y_train, X_val, y_val, X_test, y_test, df_test):
        """
        모델 성능을 상세하게 출력
        """
        self._logger.info("\n" + "="*60)
        self._logger.info("                    모델 성능 평가 결과")
        self._logger.info("="*60)
        
        # 각 데이터셋에 대한 예측
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        # NDCG 스코어 계산
        train_ndcg = ndcg_score([y_train], [y_train_pred])
        val_ndcg = ndcg_score([y_val], [y_val_pred])
        test_ndcg = ndcg_score([y_test], [y_test_pred])
        
        self._logger.info("[NDCG Score]")
        self._logger.info(f"  Train NDCG : {train_ndcg:.6f}")
        self._logger.info(f"  Valid NDCG : {val_ndcg:.6f}")
        self._logger.info(f"  Test  NDCG : {test_ndcg:.6f}")
        
        # 레이스별 Top-K 정확도 계산
        self._logger.info("[Top-K 정확도 (테스트 데이터)]")
        test_races = df_test['raceid'].unique()
        top1_correct = 0
        top2_correct = 0
        top3_correct = 0
        total_races = len(test_races)
        
        for race_id in test_races:
            race_mask = df_test['raceid'] == race_id
            race_y_true = y_test[race_mask].values
            race_y_pred = y_test_pred[race_mask]
            
            # 예측 순위 (높은 점수가 1위)
            pred_rank = np.argsort(-race_y_pred)  # 점수 높은 순으로 정렬된 인덱스
            # 실제 1위 (반전된 label에서 가장 높은 값 = 1위)
            true_winner_idx = np.argmax(race_y_true)
            
            if pred_rank[0] == true_winner_idx:
                top1_correct += 1
            if true_winner_idx in pred_rank[:2]:
                top2_correct += 1
            if true_winner_idx in pred_rank[:3]:
                top3_correct += 1
        
        self._logger.info(f"  Top-1 정확도: {top1_correct}/{total_races} ({100*top1_correct/total_races:.2f}%)")
        self._logger.info(f"  Top-2 정확도: {top2_correct}/{total_races} ({100*top2_correct/total_races:.2f}%)")
        self._logger.info(f"  Top-3 정확도: {top3_correct}/{total_races} ({100*top3_correct/total_races:.2f}%)")
        
        # 샘플 예측 결과 출력
        self._logger.info("[샘플 예측 결과 (처음 4개 레이스)]")
        sample_races = test_races[:4]
        for race_id in sample_races:
            race_mask = df_test['raceid'] == race_id
            race_y_true = y_test[race_mask].values
            race_y_pred = y_test_pred[race_mask]
            probabilities = self._ranking_scores_to_probabilities(race_y_pred)
            
            # 예측 순위 계산 (점수 높은 순 = 1위)
            pred_ranks = np.argsort(np.argsort(-race_y_pred)) + 1
            # 실제 순위 (label이 반전되었으므로 7 - label = 원래 순위)
            true_ranks = 7 - race_y_true
            
            self._logger.info(f"  Race ID: {race_id}")
            self._logger.info(f"  {'Waku':<6} {'실제순위':<10} {'예측순위':<10} {'예측점수':<12} {'예측확률':<10}")
            self._logger.info(f"  {'-'*50}")
            for i in range(len(race_y_true)):
                self._logger.info(f"  {i+1:<6} {true_ranks[i]:<10} {pred_ranks[i]:<10} {race_y_pred[i]:<12.4f} {probabilities[i]*100:<10.2f}%")
        
        self._logger.info("="*60)
    
    def _print_feature_importance(self, model, df):
        """
        Feature Importance를 정렬하여 출력
        """
        self._logger.info("\n" + "="*60)
        self._logger.info("                  Feature Importance (Top 20)")
        self._logger.info("="*60)
        
        feature = list(df.drop(columns=['class']).columns)
        importances = np.array(model.feature_importances_)
        
        df_importance = pd.DataFrame({'feature': feature, 'importance': importances})
        df_importance = df_importance.sort_values('importance', ascending=False)
        
        max_importance = df_importance['importance'].max()
        
        self._logger.info(f"{'Rank':<6} {'Feature':<20} {'Importance':<12} {'Bar'}")
        self._logger.info("-" * 60)
        
        for idx, (_, row) in enumerate(df_importance.head(20).iterrows()):
            bar_length = int(30 * row['importance'] / max_importance) if max_importance > 0 else 0
            bar = '█' * bar_length
            self._logger.info(f"{idx+1:<6} {row['feature']:<20} {row['importance']:<12.0f} {bar}")
        
        self._logger.info("="*60)
    
    def _save_model(self, model, model_filepath):
        """
        학습된 모델을 파일로 저장
        """
        self._logger.info("\n" + "="*60)
        self._logger.info("                      모델 저장")
        self._logger.info("="*60)
        
        # model_filepath가 'model_filepath' (placeholder)인 경우 자동 생성
        if model_filepath == 'model_filepath':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filepath = f"lgbm_ranker_{timestamp}.model"
        
        # 디렉토리가 없으면 생성
        model_dir = os.path.dirname(model_filepath)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
            self._logger.info(f"  디렉토리 생성: {model_dir}")
        
        # 모델 저장
        model.booster_.save_model(model_filepath)
        
        self._logger.info(f"  모델 저장 완료: {model_filepath}")
        self._logger.info(f"  파일 크기: {os.path.getsize(model_filepath) / 1024:.2f} KB")
        self._logger.info("="*60)

    def _ranking_scores_to_probabilities(self, scores) -> list[float]:
        """
        ランキングスコアを確率に変換する関数
    
        :param scores: ランキングスコアのリスト
        :return: 確率に変換されたスコアのリスト
        """
        # スコアを指数関数で変換
        exp_scores = np.exp(scores)
        
        # 正規化して確率に変換
        probabilities = exp_scores / np.sum(exp_scores)
        
        return probabilities
        
def logSetup():
    """
    loggin環境を定義する
    """ 
    prop = PropertyUtil.getInstance()
    with open(prop.getProperty('file_python_log_config'), 'r', encoding='utf-8') as f:
        log_config = json.load(f)
        
    # ファイル名をタイムスタンプで作成
    log_file_name = prop.getProperty('file_python_log')
    log_config["handlers"]["fileHandler"]["filename"] = log_file_name.format(datetime.utcnow().strftime("%Y%m%d"))

    config.dictConfig(log_config)
    
def main(argv):
    # argv: {params} {csv_file_path} {model_file_path} {feature_name_list} {feature_type_list}
    # ex: boosting_type=gbdt,learning_rate=0.1 D:/Dev/experiment/expr10/arff/9997_rank1.csv D:/Dev/experiment/expr10/evaluation/model_release/9997/1/nopattern/9997_nopattern_yyyymmdd_rank1.model nw1,nw2,nw3,nw4,nw5,nw6,class float,float,float,float,float,float,category
    
    #param len check
    if len(argv) < 6:
        logger = getLogger('server')
        logger.error('Usage: python xxxModelGenerator.py {params} {csv_file_path} {model_file_path} {feature_name_list} {feature_type_list}')
        return -1
    
    prop = PropertyUtil.getInstance()
    prop.addFile('C:/Dev/github/pod_boatrace/properties/expr10/expr10.properties')
    
    # loggin set up
    logSetup()

    tester = BoatLGBMRankerTest()
    tester.execute(argv[1], argv[2], argv[3], argv[4], argv[5])
    
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))

