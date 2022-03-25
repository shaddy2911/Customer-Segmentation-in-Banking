import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN,SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score


st.title('Customer Segmentation')
image=Image.open('LinkedIn Cover.jpg')
st.image(image,use_column_width=True)

def main():
    data=st.file_uploader('Upload The Dataset',type=['csv'])
    if data is not None:
        st.success('Data Successfully Uploaded')
    if data is not None:
        df=pd.read_csv(data)
        st.write('Raw Data',df)
        creditcard_df = df
        
        #st.write(creditcard_df.isnull().sum())
        # fill mean value in place of missing values
        creditcard_df["MINIMUM_PAYMENTS"] = creditcard_df["MINIMUM_PAYMENTS"].fillna(creditcard_df["MINIMUM_PAYMENTS"].mean())
        creditcard_df["CREDIT_LIMIT"] = creditcard_df["CREDIT_LIMIT"].fillna(creditcard_df["CREDIT_LIMIT"].mean())
        #st.write(creditcard_df.isnull().sum())

        concat_1=creditcard_df['CUST_ID']
        #st.write(concat_1)

        # drop unnecessary columns
        creditcard_df.drop(columns=["CUST_ID"],axis=1,inplace=True)

        # find outlier in all columns
        for i in creditcard_df.select_dtypes(include=['float64','int64']).columns:
            max_thresold = creditcard_df[i].quantile(0.95)
            min_thresold = creditcard_df[i].quantile(0.05)
            creditcard_df_no_outlier = creditcard_df[(creditcard_df[i] < max_thresold) & (creditcard_df[i] > min_thresold)].shape
            #st.write((" outlier in ",i,"is" ,int(((creditcard_df.shape[0]-creditcard_df_no_outlier[0])/creditcard_df.shape[0])*100),"%"))
        # remove outliers from columns having nearly 10% outlier
        max_thresold_BALANCE = creditcard_df["BALANCE"].quantile(0.95)
        min_thresold_BALANCE = creditcard_df["BALANCE"].quantile(0.05)
        max_thresold_CREDIT_LIMIT = creditcard_df["CREDIT_LIMIT"].quantile(0.95)
        min_thresold_CREDIT_LIMIT = creditcard_df["CREDIT_LIMIT"].quantile(0.05)
        max_thresold_PAYMENTS = creditcard_df["PAYMENTS"].quantile(0.95)
        min_thresold_PAYMENTS = creditcard_df["PAYMENTS"].quantile(0.05)
        creditcard_df_no_outlier = creditcard_df[(creditcard_df["CREDIT_LIMIT"] < max_thresold_CREDIT_LIMIT) & (creditcard_df["CREDIT_LIMIT"] > min_thresold_CREDIT_LIMIT) & (creditcard_df["BALANCE"] < max_thresold_BALANCE) & (creditcard_df["BALANCE"] > min_thresold_BALANCE) &  (creditcard_df["PAYMENTS"] < max_thresold_PAYMENTS) & (creditcard_df["PAYMENTS"] > min_thresold_PAYMENTS)]
        #st.write(creditcard_df_no_outlier.head())
        # scale the DataFrame
        scalar=StandardScaler()
        creditcard_scaled_df = scalar.fit_transform(creditcard_df)

        # convert the DataFrame into 2D DataFrame for visualization
        pca = PCA(n_components=2)
        principal_comp = pca.fit_transform(creditcard_scaled_df)
        pca_df = pd.DataFrame(data=principal_comp,columns=["pca1","pca2"])
        #st.write(pca_df.head())


        # find 'k' value by Elbow Method
        inertia = []
        range_val = range(1,15)
        for i in range_val:
            kmean = KMeans(n_clusters=i)
            kmean.fit_predict(pd.DataFrame(creditcard_scaled_df))
            inertia.append(kmean.inertia_)
        plt.plot(range_val,inertia,'bx-')
        plt.xlabel('Values of K') 
        plt.ylabel('Inertia') 
        plt.title('The Elbow Method using Inertia')
        st.set_option('deprecation.showPyplotGlobalUse', False) 
        #st.pyplot()

        # apply kmeans algorithm
        kmeans_model=KMeans(n_clusters=4, random_state=123)
        kmeans_model.fit_predict(creditcard_scaled_df)
        idx = np.argsort(kmeans_model.cluster_centers_.sum(axis=1))
        lut = np.zeros_like(idx)
        lut[idx] = np.arange(4)
         
        pca_df_kmeans= pd.concat([pca_df,pd.DataFrame({'cluster':lut[kmeans_model.labels_]})],axis=1)
        # visualize the clustered dataframe
        # Scatter Plot
        plt.figure(figsize=(10,10))
        #palette=['dodgerblue','red','green','blue','black','pink','gray','purple','coolwarm']
        ax=sns.scatterplot(x="pca1",y="pca2",hue="cluster",data=pca_df_kmeans,palette=['red','green','blue','black'])
        plt.title("Clustering using K-Means Algorithm")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.write('Cluster Segmentation') 
        st.pyplot()

        # find all cluster centers
        cluster_centers = pd.DataFrame(data=kmeans_model.cluster_centers_,columns=[creditcard_df.columns])
        # inverse transfor the data
        cluster_centers = scalar.inverse_transform(cluster_centers)
        cluster_centers = pd.DataFrame(data=cluster_centers,columns=[creditcard_df.columns])
        #cluster_centers
        
        # create a column as "cluster" & store the respective cluster name that they belongs to
        creditcard_cluster_df = pd.concat([creditcard_df,pd.DataFrame({'cluster':lut[kmeans_model.labels_]})],axis=1)
        #st.write(creditcard_cluster_df.head())


        cluster_1_df = creditcard_cluster_df[creditcard_cluster_df["cluster"]==0]
        #st.subheader('Cluster-1')
        #st.write(cluster_1_df.head())
        #st.write(cluster_1_df.describe())

        cluster_2_df = creditcard_cluster_df[creditcard_cluster_df["cluster"]==1]
        #st.subheader('Cluster-2')
        #st.write(cluster_2_df.head())

        cluster_3_df = creditcard_cluster_df[creditcard_cluster_df["cluster"]==2]
        #st.subheader('Cluster-3')
        #st.write(cluster_3_df.head())

        cluster_4_df = creditcard_cluster_df[creditcard_cluster_df["cluster"] == 3]
        #st.subheader('Cluster-4')
        #st.write(cluster_4_df.head())

        #st.write(creditcard_cluster_df.head())

        df=[concat_1,creditcard_cluster_df]
        result = pd.concat((df),axis=1)
        #st.write(result.head())
        #o=result['CUST_ID'].unique()
        #id=st.selectbox('Select The Customer ID:',o)
        #gk=result.groupby('CUST_ID'	)
        #f_o=gk.get_group(id)
        #st.write(f_o[['CUST_ID','cluster']])
        #cl = f_o["cluster"].values.tolist()
        #print(cl)
        #if 0  in list(cl):
            #st.write(""".
                        
                        #a)	These are the people who have the highest balance in 
                            #Their credit card account after using it for one financial cycle
                        
                        #b)	 These people don’t have very frequent purchases or even don’t buy expensive things with their credit card as their amount of purchases and single shot purchase amount are relatively low 
                        
                        #c)	There are also very few installments aligned with these people as their advance payments are high,
                            #This also shows that these people are not very fond of liabilities
                        
                        #d)	These people have high credit limit which shows that they have a decently good income.
                        
                        #Recommendation:- Very good market for pitching savings account, a savings account with good interest rate have a very good chance of interesting them""")
        
        #if 1 in list(cl):
            #st.write(""".
                        
                        #a)	These are the people with lowest of the credit card account balance.
                        
                        #b)	Their frequency of purchases is high, which shows they use their credit card quite frequent 
                        
                        #c)	Their one-off purchases are not very high which shows they use their credit card often for daily needs
                        
                        #d)	Their purchase frequency is high and they also have high installment purchase frequency which shows they buy things on installments and don’t object on creating a liability
                        
                        #e)	Their credit card limit is comparatively less but they have a good full payment percentage score, which shows they can use the loan money and can also return the money giving banks good business

                        #Recommendation:- A good market for pitching savings account and even loan account but since they do have low credit limit their income isn’t very high, thus a loan of less amount is recommended to this cluster""")
        
        #if 2 in list(cl):
            #st.write(""".
                        
                        #a)	Very passive cluster of customers, hardly use credit card as their amount and frequency of purchases is very low, the lowest among all the clusters,
                        
                        #b)	Very low credit limit, which shows not very good income.
                        
                        #c)	Did not pay back amount of whatever purchases they did with their credit card, thus not very good chances of paying back loans either.

                        #Recommendation:- Not a  good market for pitching savings account or  even loan account as they are very passive in nature""")
        
        #if 3 in list(cl):
            #st.write(""".
                        
                        #a)	High amount of credit card balance
                        
                        #b)	High amount of credit limit, which means high income,
                        
                        #c)	Very high amount of purchases
                        
                        #d)	Frequency of purchases is also very high, can be categorized as compulsive buyers 
                        
                        #e)	They have high installments aligned with their card,
                        #But their full amount payback is very good and their credit limit is the highest among all, which makes them ideal candidate for pitching loans, as the chances of them taking one are high and they can also pay back the loan because of their high income, which gives banks a very good business

                        #Recommendation:- Very good market for pitching  a Loan account, compulsive buyers by nature, Hot leads for pitching loans""")
        

        if st.button('Feature Importance'):
            one_hot_encoded_data = pd.get_dummies(creditcard_cluster_df, columns = ['cluster'])
            df=pd.DataFrame(one_hot_encoded_data)
            X=df[['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES','CASH_ADVANCE','PURCHASES_FREQUENCY','ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY','CASH_ADVANCE_FREQUENCY','CASH_ADVANCE_TRX','PURCHASES_TRX','CREDIT_LIMIT','PAYMENTS','MINIMUM_PAYMENTS','PRC_FULL_PAYMENT','TENURE']]
            y=df['cluster_0'] 
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            clf=RandomForestClassifier(n_estimators=100)
            clf.fit(X_train,y_train)
            y_pred=clf.predict(X_test)
            y_pred=pd.DataFrame(y_pred)
            #st.write(y_pred.head())
            st.title('Feature Importance Of Cluster 0')
            def plot_importance(model, features, num=len(X), save=None):
                feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
                plt.figure(figsize=(8, 3))
                sns.set(font_scale=1)
                sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:5])
                plt.title('Cluster_0')
                plt.tight_layout()
                plt.show()
                #st.pyplot()
                if save:
                    plt.savefig('importances.png')

            model = RandomForestClassifier()
            model.fit(X,y)
            st.pyplot(plot_importance(model, X))

            #Cluster 2
            st.title('Feature Importance Of Cluster 1')
            X=df[['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES','CASH_ADVANCE','PURCHASES_FREQUENCY','ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY','CASH_ADVANCE_FREQUENCY','CASH_ADVANCE_TRX','PURCHASES_TRX','CREDIT_LIMIT','PAYMENTS','MINIMUM_PAYMENTS','PRC_FULL_PAYMENT','TENURE']]
            y=df['cluster_1'] 
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            clf=RandomForestClassifier(n_estimators=100)
            clf.fit(X_train,y_train)
            y_pred=clf.predict(X_test)
            y_pred=pd.DataFrame(y_pred)
            def plot_importance(model, features, num=len(X), save=None):
                feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
                plt.figure(figsize=(8, 3))
                sns.set(font_scale=1)
                sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:5])
                plt.title('Cluster_1')
                plt.tight_layout()
                plt.show()
                #st.pyplot()
                if save:
                    plt.savefig('importances.png')

            model = RandomForestClassifier()
            model.fit(X,y)
            st.pyplot(plot_importance(model, X))

            #Cluster 2
            st.title('Feature Importance Of Cluster 2')
            X=df[['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES','CASH_ADVANCE','PURCHASES_FREQUENCY','ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY','CASH_ADVANCE_FREQUENCY','CASH_ADVANCE_TRX','PURCHASES_TRX','CREDIT_LIMIT','PAYMENTS','MINIMUM_PAYMENTS','PRC_FULL_PAYMENT','TENURE']]
            y=df['cluster_2'] 
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            clf=RandomForestClassifier(n_estimators=100)
            clf.fit(X_train,y_train)
            y_pred=clf.predict(X_test)
            y_pred=pd.DataFrame(y_pred)
            def plot_importance(model, features, num=len(X), save=None):
                feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
                plt.figure(figsize=(8, 3))
                sns.set(font_scale=1)
                sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:5])
                plt.title('Cluster_2')
                plt.tight_layout()
                plt.show()
                #st.pyplot()
                if save:
                    plt.savefig('importances.png')

            model = RandomForestClassifier()
            model.fit(X,y)
            st.pyplot(plot_importance(model, X))

            #Cluster 2
            st.title('Feature Importance Of Cluster 3')
            X=df[['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES','CASH_ADVANCE','PURCHASES_FREQUENCY','ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY','CASH_ADVANCE_FREQUENCY','CASH_ADVANCE_TRX','PURCHASES_TRX','CREDIT_LIMIT','PAYMENTS','MINIMUM_PAYMENTS','PRC_FULL_PAYMENT','TENURE']]
            y=df['cluster_3'] 
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            clf=RandomForestClassifier(n_estimators=100)
            clf.fit(X_train,y_train)
            y_pred=clf.predict(X_test)
            y_pred=pd.DataFrame(y_pred)
            def plot_importance(model, features, num=len(X), save=None):
                feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
                plt.figure(figsize=(8, 3))
                sns.set(font_scale=1)
                sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:5])
                plt.title('Cluster_3')
                plt.tight_layout()
                plt.show()
                #st.pyplot()
                if save:
                    plt.savefig('importances.png')

            model = RandomForestClassifier()
            model.fit(X,y)
            st.pyplot(plot_importance(model, X))
            
        
        if st.button('Explore Clusters'):
            st.title('Explore Clusters')
            st.write('-'*50)
            #st.title('Cluster 0')
            #c=['Cluster_0','Cluster_1','Cluster_2','Cluster_3']
            #cb=st.multiselect('Select The Clusters To Explore',list(c))
            #if cb=='Cluster_0':
            st.title('Cluster_0')
            cluster_1_df.reset_index(drop=True, inplace=True)
            k=[concat_1,cluster_1_df]
            result1=pd.concat((k),axis=1)
            st.write(result1)
            d=(result1.describe().T)
            fd=d.rename(columns={'50%':'Cluster_0-Median','mean':'Cluster_0-Mean','std':'Cluster_0-STD'},inplace=False)
            s=(fd[['Cluster_0-Mean','Cluster_0-STD','Cluster_0-Median']])
            #st.write('-'*50)
            #st.title('Cluster 1')
            #if cb=='Cluster_1':
            st.title('Cluster 1')
            cluster_2_df.reset_index(drop=True, inplace=True)
            k1=[concat_1,cluster_2_df]
            result2=pd.concat((k1),axis=1)
            st.write(result2)
            d1=(result2.describe().T)
            fd1=d1.rename(columns={'50%':'Cluster_1-Median','mean':'Cluster_1-Mean','std':'Cluster_1-STD'},inplace=False)
            s1=(fd1[['Cluster_1-Mean','Cluster_1-Median','Cluster_1-STD']])
            st.write('-'*50)
            #if cb=='Cluster_2':
            #st.title('Cluster 2')
            cluster_3_df.reset_index(drop=True, inplace=True)
            k2=[concat_1,cluster_3_df]
            result3=pd.concat((k2),axis=1)
            st.title('Cluster 2')
            st.write(cluster_3_df)
            d2=(result3.describe().T)
            fd2=d2.rename(columns={'50%':'Cluster_2-Median','mean':'Cluster_2-Mean','std':'Cluster_2-STD'},inplace=False)
            s2=(fd2[['Cluster_2-Mean','Cluster_2-Median','Cluster_2-STD']])
            st.write('-'*50)
            #st.title('Cluster 3')
            #if cb=='Cluster_3':
            st.title('Cluster 3')
            cluster_4_df.reset_index(drop=True, inplace=True)
            k3=[concat_1,cluster_4_df]
            result4=pd.concat((k3),axis=1)
            st.write(result4)
            d3=(result4.describe().T)
            fd3=d3.rename(columns={'50%':'Cluster_3-Median','mean':'Cluster_3-Mean','std':'Cluster_3-STD'},inplace=False)
            s3=(fd3[['Cluster_3-Mean','Cluster_3-Median','Cluster_3-STD']])
            c1=[s,s1,s2,s3]
            r=pd.concat((c1),axis=1)
            r1=r[['Cluster_0-Mean','Cluster_1-Mean','Cluster_2-Mean','Cluster_3-Mean','Cluster_0-Median','Cluster_1-Median','Cluster_2-Median','Cluster_3-Median','Cluster_0-STD','Cluster_1-STD','Cluster_2-STD','Cluster_3-STD']]
            st.write(r1)
            


        
        if st.button('Product-Persona Mapping'):
            data=pd.read_csv('C:/Users/Shaddy/Desktop/projects/streamlit/Product_Persona_Mapping.csv')
            st.write(data)
            st.write(""".
                        
                        Loan:- probability of cluster_0 is 8%

                        Saving:- probability of cluster_0 is 10%

                        Loan:- probability of cluster_1 is 58%

                        Saving:- probability of cluster_1 is 35%

                        Loan:- probability of cluster_2 is 70%

                        Saving:- probability of cluster_2 is 76%

                        Loan:- probability of cluster_3 is 38%

                        Saving:- probability of cluster_3 is 30%""")
        #if st.button('View Cluster_0'):
            #st.write(result1)
        #if st.button('View Cluster_1'):
            #st.write(result2)
        #if st.button('View Cluster_2'):
            #st.write(result3)
        #if st.button('View Cluster_3'):
            #st.write(result4)



        o=result['CUST_ID'].unique()
        id=st.selectbox('Select The Customer ID:',o)
        gk=result.groupby('CUST_ID'	)
        f_o=gk.get_group(id)
        st.write(f_o[['CUST_ID','cluster']])
        cl = f_o["cluster"].values.tolist()
        #st.write(f_o[['CUST_ID','cluster']])
        cl = f_o["cluster"].values.tolist()
        #print(cl)
        if 2  in list(cl):
            st.write("""Recommendation:-  Savings Account
                        
                        a)	 high balance
                            
                        
                        b)	 very frequent purchases 
                        
                        c)	 very few installments aligned with thier cards
                            
                        
                        d)	These people have high credit limit 
                        
                        """)
        
        if 1 in list(cl):
            st.write(""" Recommendation:- Loan Account  Probably With High Interest Rates
                        
                        a)	Lowest balance.
                        
                        b) frequency of purchases is high 
                        
                        c) one-off purchases are not very high
                         
                        
                        e)	Low Credit Card Limit

                        """)
        
        if 0 in list(cl):
            st.write("""Recommendation:- Not Good For Both Loan And Savings Acccount
                        
                        a)	Very passive cluster of customers, hardly use credit card 
                        
                        b)	Very low credit limit
                        
                        c)	Full Payment score not good.

                        """)
        
        if 3 in list(cl):
            st.write("""Recommendation:- Ideal For Loan Account
                        
                        a)	High  credit card balance
                        
                        b)	High  credit limit
                        
                        c)	Very high amount of purchases
                        
                        d)	Frequency of purchases is also very high 
                        
                        e)	They have high installments aligned with their card
                        

                        """)
        #if st.button('Product-Inference Table'):
            #data=pd.read_csv('C:/Users/Shaddy/Desktop/projects/streamlit/Product_Inference_Table.csv')
            #st.write(data)
        





        





                                                                                                                                                         

        

        











if __name__ == '__main__':
    main()