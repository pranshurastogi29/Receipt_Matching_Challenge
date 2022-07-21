import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import *
import seaborn as sns
import pandas as pd
import numpy as np
import seaborn as sns
import pickle as pkl
import joblib
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

def data_creation(path, mode, output_dir = './'):
    df = pd.read_csv(path)
    l = []
    for i in df.values:
        l.append(i[-1].split(":"))
    df = df.reset_index()
    df2 = pd.DataFrame(l)
    df1 = df[['level_0','level_1','level_2']]
    df1.columns = df.columns[-1].split(":")[:3]
    df2.columns = df.columns[-1].split(":")[3:]
    df3 = pd.concat([df1,df2], axis = 1)
    df3['company_id'] = df3['company_id'].apply(lambda x : x.split(":")[0])
    df3['matched_transaction_id'] = df3['matched_transaction_id'].apply(lambda x : x.split(":")[0])
    
    if mode == 'train':
        df4 = df3[df3['matched_transaction_id']==df3['feature_transaction_id']]
        df4['class'] = 1
        df5 = df3[df3['matched_transaction_id']!=df3['feature_transaction_id']]
        df5['class'] = 0
        for i in df3.columns:
            df3[i] = df3[i].astype(float)
            df4[i] = df4[i].astype(float)
            df5[i] = df5[i].astype(float)
        df6 = pd.concat([df4,df5]).reset_index(drop = True)
        df6.to_csv(output_dir+'/train_data.csv', index = False)
        return output_dir+'/train_data.csv'
        
    if mode == 'test':
        for i in df3.columns:
            df3[i] = df3[i].astype(float)
        df3.to_csv(output_dir+'/test_data.csv', index = False)
        return output_dir+'/test_data.csv'


def feature_creation(path, mode, output_dir = './'):
    
    df6 = pd.read_csv(path)
    feats = list(df6.columns)
    if mode == 'train':
        feats.remove('class')
    else:
        pass
    
    df6[feats] = df6[feats] + 1
    df6 = df6.drop(columns = ['receipt_id','company_id','matched_transaction_id','feature_transaction_id','DifferentPredictedDate', 'DifferentPredictedTime'])
    feats = ['DateMappingMatch','AmountMappingMatch','DescriptionMatch','TimeMappingMatch','PredictedNameMatch',
     'ShortNameMatch','PredictedAmountMatch','PredictedTimeCloseMatch']
    numerical_cols_original = ['DateMappingMatch','PredictedNameMatch','DescriptionMatch']
    categorical_cols_original = ['AmountMappingMatch','TimeMappingMatch','PredictedAmountMatch','ShortNameMatch','PredictedTimeCloseMatch']
    
    poly_feats = PolynomialFeatures(degree = 3, interaction_only = True)
    total_data = pd.DataFrame(poly_feats.fit_transform(df6[feats]))
    total_data.columns = poly_feats.get_feature_names_out(feats)
    
    df7 = total_data.drop(columns = ['1'])
    
    feats = ['DateMappingMatch', 'AmountMappingMatch', 'DescriptionMatch','TimeMappingMatch', 'PredictedNameMatch', 'ShortNameMatch','PredictedAmountMatch', 'PredictedTimeCloseMatch']
    for i in feats:
        feats2work = feats
        feats2work.remove(i)
        for j in feats2work:
            finals = df6.groupby(i)[j].agg(['mean', 'std', 'min', 'skew', 'count', 'nunique','median','cumsum'])
            finals.columns = [i + '_' + j + '_' + k for k in finals.columns]
            df6 = df6.join(finals, on = i)
            
    columns_a = ['DateMappingMatch', 'DescriptionMatch', 'TimeMappingMatch', 'ShortNameMatch', 'PredictedAmountMatch']
    columns_b = ['AmountMappingMatch', 'PredictedTimeCloseMatch', 'PredictedNameMatch']

    for col_a in columns_a:
        for col_b in columns_b:
            for df in [df6]:
                df[f'{col_a}_div_mean_{col_b}'] = df[col_a] / df.groupby([col_b])[col_a].mean()
                df[f'{col_a}_div_std_{col_b}'] = df[col_a] / df.groupby([col_b])[col_a].std()
                df[f'{col_a}_div_median_{col_b}'] = df[col_a] / df.groupby([col_b])[col_a].median()
                df[f'{col_a}_div_var_{col_b}'] = df[col_a] / df.groupby([col_b])[col_a].count()
                df[f'{col_a}_div_max_{col_b}'] = df[col_a] / df.groupby([col_b])[col_a].max()
                df[f'{col_a}_div_min_{col_b}'] = df[col_a] / df.groupby([col_b])[col_a].min()
            
    for col_a in columns_a:
        for col_b in columns_b:
            for df in [df6]:
                df[f'{col_a}_sub_mean_{col_b}'] = df[col_a]   - df.groupby([col_b])[col_a].mean()
                df[f'{col_a}_sub_std_{col_b}'] = df[col_a]    - df.groupby([col_b])[col_a].std()
                df[f'{col_a}_sub_median_{col_b}'] = df[col_a] - df.groupby([col_b])[col_a].median()
                df[f'{col_a}_sub_var_{col_b}'] = df[col_a]    - df.groupby([col_b])[col_a].var()
                df[f'{col_a}_sub_max_{col_b}'] = df[col_a]    - df.groupby([col_b])[col_a].max()
                df[f'{col_a}_sub_min_{col_b}'] = df[col_a]    - df.groupby([col_b])[col_a].min()
                
    feats = ['DateMappingMatch', 'AmountMappingMatch', 'DescriptionMatch','TimeMappingMatch', 'PredictedNameMatch', 'ShortNameMatch','PredictedAmountMatch', 'PredictedTimeCloseMatch']
    for i in feats:
        feats2work = feats
        feats2work.remove(i)
        for j in feats2work:
            df6[i+'div'+j] = df6[i]/df6[j]
            df6[i+'mul'+j] = df6[i]*df6[j]
            df6[i+'add_div'+j] = (df6[i]+df6[j])/df6[i]
            df6[i+'sub_mul'+j] = (df6[i]-df6[j])*df6[i]
            
    df7 = df7.drop(columns = ['DateMappingMatch', 'AmountMappingMatch', 
                          'DescriptionMatch','TimeMappingMatch', 'PredictedNameMatch', 
                          'ShortNameMatch','PredictedAmountMatch', 'PredictedTimeCloseMatch'])
    df6 = pd.concat([df7, df6], axis = 1)
    df6 = df6.fillna(0)
    df6.replace([np.inf, -np.inf], 0, inplace=True)
    df6.to_csv(output_dir+'/final_features'+mode+'.csv',index = False)
    return output_dir+'/final_features'+mode+'.csv'


def train_model(path = '../input/tide-match-data/data_interview_test.csv', output_dir = './'):
    
    train_data_path = data_creation(path, mode = 'train')
    train_data_path = feature_creation(train_data_path , mode = 'train')
    
    df6 = pd.read_csv(train_data_path)
    feats = list(df6.columns)
    feats.remove('class')
    final_preds = 0
    x1 = df6[feats]
    y1 = df6['class']*1
    pca = PCA(n_components=50, random_state = 42)
    feature_cols = feats
    X_pca = pca.fit_transform(df6[feature_cols])

    X_pca = pd.DataFrame(X_pca, columns=['pca_'+ str(i) for i in range(50)], index=df6.index)

    df6 = pd.concat([X_pca], axis=1)

    x1 = df6
    feats = list(df6.columns)
    x_train, x_test, y_train,y_test = train_test_split(x1,y1, test_size=0.2,random_state=42, stratify = y1)    

    x_train, y_train = SMOTE(sampling_strategy = .4, random_state=42).fit_resample(x_train, y_train)
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 42)
    x_train = x1.reset_index(drop = True)
    y_train = y1.reset_index(drop = True)
    fold = 0
    for train_index, test_index in skf.split(x_train, y_train):
        x_tr, x_tt = x_train.values[train_index], x_train.values[test_index]
        y_tr, y_tt = y_train[train_index], y_train[test_index]
        x_tr = pd.DataFrame(x_tr)
        x_tr.columns = feats
        x_test = pd.DataFrame(x_test)
        x_test.columns = feats
        clf = XGBClassifier(random_state = 42).fit(x_tr, y_tr)
        pred = clf.predict_proba(x_test)
        final_preds = final_preds + pred/10
        joblib.dump(clf, 'xgb_classifier'+str(fold)+'.pkl')
        fold = fold + 1
    
    print(classification_report(y_test, np.argmax(final_preds, axis = 1)))
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
    plot_precision_recall_curve(clf,x_test, y_test,ax=ax1)
    skplt.metrics.plot_roc_curve(y_test, final_preds)
    sns.heatmap(confusion_matrix(y_test, np.argmax(final_preds, axis = 1)),annot=True, fmt='d',cmap='Reds', ax=ax2,)
    ax1.set_xlabel('Pred')
    ax1.set_ylabel('True')
    plt.show()

    pkl.dump(pca, open(output_dir+"/pca.pkl","wb"))


def prediction_pipeline(path = '../input/tide-match-data/data_interview_test.csv', pca_path = 'pca.pkl', model_path = 'xgb_classifier1.pkl', output_dir = './'):
    
    test_data_path = data_creation(path, mode = 'test')
    test_data_path = feature_creation(test_data_path , mode = 'test')
    df6 = pd.read_csv(test_data_path)
    feats = list(df6.columns)
    pca = pkl.load(open(pca_path,'rb'))
    X_pca = pca.transform(df6[feats])

    X_pca = pd.DataFrame(X_pca, columns=['pca_'+ str(i) for i in range(50)], index=df6.index)
    model = joblib.load(model_path)
    preds_proba = model.predict_proba(X_pca)[:,1]
    #print(sum(preds_proba>.3), sum(preds_proba>.2), sum(preds_proba>.5))
    return preds_proba