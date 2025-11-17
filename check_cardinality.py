import pandas as pd

feature_names = "mm,jyo,race,turn,grade,raty,femcnt,alvt,time,fixent,en1,en2,en3,en4,en5,en6,nw1,nw2,nw3,nw4,nw5,nw6,n2w1,n2w2,n2w3,n2w4,n2w5,n2w6,n3w1,n3w2,n3w3,n3w4,n3w5,n3w6,lw1,lw2,lw3,lw4,lw5,lw6,l2w1,l2w2,l2w3,l2w4,l2w5,l2w6,l3w1,l3w2,l3w3,l3w4,l3w5,l3w6,m2w1,m2w2,m2w3,m2w4,m2w5,m2w6,sex1,sex2,sex3,sex4,sex5,sex6,lv1,lv2,lv3,lv4,lv5,lv6,age1,age2,age3,age4,age5,age6,weit1,weit2,weit3,weit4,weit5,weit6,fly1,fly2,fly3,fly4,fly5,fly6,late1,late2,late3,late4,late5,late6,avgst1,avgst2,avgst3,avgst4,avgst5,avgst6,class".split(',')
feature_types = "category,category,category,category,category,category,int,int,float,float,category,category,category,category,category,category,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,category,category,category,category,category,category,category,category,category,category,category,category,int,int,int,int,int,int,float,float,float,float,float,float,int,int,int,int,int,int,int,int,int,int,int,int,float,float,float,float,float,float,category".split(',')

df = pd.read_csv('C:/Dev/experiment/expr10/arff/39104_rank1.csv', header=None)
print('Shape:', df.shape)
print('\n=== All Categorical Features ===')
for i in range(len(df.columns)):
    unique_count = df[i].nunique()
    if feature_types[i] == 'category':
        marker = ' *** HIGH' if unique_count > 100 else ''
        print(f'Col {i:3d} ({feature_names[i]:8s}) [category]: {unique_count:5d} unique values{marker}')
