import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier#knn
from sklearn.ensemble import RandomForestClassifier#random forest
from sklearn.model_selection import train_test_split

df=pd.read_csv('D:\Coduri\pbi\loan_train_data.csv')

"""
print(df)
dataframe 
#print(df['ApplicantIncome']); ->ne da un set de valori pe care putem sa aplicam niste operatii vectorizate
#coloana aceasta ar trebi filtrata,astfel aplicam o operatie la nivel de coloana
#putem sa aplicam o operatie astfel
print(df['ApplicantIncome'] * 2)
putem aplica si operatori logici
print(df["ApplicantIncome"] > 5000)
Sau putem afisa indexul datelor care satisfac o conditie astfel
print(df["ApplicantIncome"].loc[df["ApplicantIncome"] > 5000])


Pentru a prelucra datele
->identificam coloanele care pot fi tratate ca categorii,identificam coloanele numerice si
daca trebiue sa aplicam ceva anume,iar apoi le punem intr-un model.
Coloane categorii:gender,married,education,self-employed si proprietyArea. Ultima coloana 
o lasam separat deoarece pe aceea o vom estima(le lasam doar pe cele pe care le avem la dispozitie irl)

print(df['Property_Area'].unique()) vedem cate valori unice sunt intr-o coloana
print(df['Property_Area'].value_counts())  si cate sunt din fiecare,cat de echilibrate sunt
plt.bar([0,1,2],df["Property_Area"].value_counts())
plt.show()
plt.hist(df['ApplicantIncome'],bins=100);
plt.show();

->pentru prelucrare efectiva,ne intereseaza sa transformam totul in numere. Luam toate coloanele 
cateogrice si creem coloane separate cu valori numerice
"""

#print(df["Gender"].unique())
df['GenderEncoded']= 0 
df.loc[df["Gender"]=='Male',"GenderEncoded"]=2
df.loc[df["Gender"]=='Female',"GenderEncoded"]=1
df.drop(columns=['Loan_ID','Gender'],inplace=True)
#LoanID nu are nicio valoare,este doar un ID si poate fi scos,la fel si 
# gender deoarece a fost procesat deja
#normal acest drop face doar o referinta,asa ca avem nevoie de inplace=True
df["Married"]=df['Married'].map({'No':0,'Yes':1})
df['Dependents']=df['Dependents'].map({'0':0,'1':1,'2':2,'3+':3})
df['Education']=df['Education'].map({'Not Graduate':0,'Graduate':1})
#daca atribuim 1,2,3 ca valori pentru Property_Area,unele modele s-ar putea sa intrepreteze prost datele
#adica sa vada ca 1<3 si sa ia aceasta conditie ca una necesara pentru rezultat.Recomandat este sa 
#sa separam valorile in coloane si sa atribuim True sau False(Coloana Rural are True daca traieste la tara
# si False daca nu)
df=pd.get_dummies(df,columns=['Property_Area'])#adauga la final cate o coloana cu parametrii din Property_Area
#(Property_Area_Rural,Property_Area_Urban,Property_Area_Semiurban)
#tipul de procesari la restul coloanelor se leaga de cate valori lipsa avem
#modelul o sa ignore template-urile cu valori lipsa.
#print(df.info()) ne spune nuamrul valorilor prezente din care reiese si numarul
#de valori lipsa. Avem valori lipsa la LoanAmmount etc
"""
Vom umple valorile lipsa cu media celorlalte valori! ->inputare
Mai putem sa le eliminam complet,dar daca avem putine teste nu este recomandat(spre exemplu aici)
"""
df.loc[df['LoanAmount'].isna(),'LoanAmount'] = df['LoanAmount'].mean()
df.loc[df['Loan_Amount_Term'].isna(),'Loan_Amount_Term'] = df['Loan_Amount_Term'].mean()
df.loc[df['Credit_History'].isna(),'Credit_History'] = df['Credit_History'].mean()

"""
Vom folosi algoritmul K Nearest Neighbors
Fiecare rand este ca un vector. 
Sa ne imaginam ca avem un set de date cu 2 dimensiuni. Fiecare set este ca un punct(coordonatele
pe fiecare dimensiune). Avem posibilitatea de a estima distanta intre cele doua puncte(radicalul diferentei patratelor)
Setul nostru de date contine doua tipuri de puncte,spre exemplu cele rosii din prima clasa
si cele albstre din a doua clasa. Cand apare un punct nou,pe care modelul incearca sa il clasifice
algoritmul KNN se uita la cei mai apropiati k vecini(de exemplu daca k=3 si cei mai apropiati vecini
sunt albastri,atunci si punctul nou va fi albastru => punctul va fi clasificat ca albastru).
Chiar daca noi avem mai multe dimensiuni,pricipiul de calculat distante va fi similar
Algoritmul o sa se uite la toti parametrii(gender,married,loanamount etc) si o sa vada
care sunt Y si care sunt N. Cand o sa ii dam un punct nou de clasificat,el o sa se uite la cele mai 
apropiate k puncte si majoritate dintre acesti vecini o sa dicteze clasa noului punct.
Outlieri=puncte rosii/verzi car se afla la extrema opusa din setul de date(ex:punct rosu care se afla in zona
celor albastre,dar asa a fost dat de input). Este recomandat sa fie scoase.
"""

"""
https://scikit-learn.org/stable/ - pentru mai multe informatii
"""
#coloanele de input/interes
X = df[ [
    'Married',
    'Dependents',
    'Education',
    'ApplicantIncome',
    'CoapplicantIncome',
    'LoanAmount',
    'Loan_Amount_Term',
    'Credit_History',
    'Property_Area_Rural',
    'Property_Area_Semiurban',
    'Property_Area_Urban'
] ]
#ne da cel mai comun element
#localizam randurile unde X[depenents] sunt nule si le punem pe coloana dependents
X.loc[X['Dependents'].isna(),'Dependents'] = X['Dependents'].mode()[0]
X.loc[X['Married'].isna(),'Married'] = X['Married'].mode()[0]
#print(X.info())observam ca avem cateva valori lipsa la dependents
#coloana de output
y=df['Loan_Status'].map({'Y':1,'N':0})

model=KNeighborsClassifier()

model.fit(X,y)

print(model.score(X,y))

"""
Vom folosi si modelul Random Forest
Este un model special,ia setul de date,aplica o anumita decizie pe un set de parametri random
Si vede cum pune conditia dintre acestia(<,> sau =) pentru a pune termenii cat mai 
echilibrat in raport cu clasele. Dupa imparte aceste seturi,mai face o decizie si tot asa,pana 
cand ramane un singur set la final(astfel vom avea un arbore binar de decizie)
Un random forest ia 100 de arbori de devizie,ii aplica pe toti si se foloseste de votul majoritar
pentru a-si da sama de clasa obiectului rezultat. Avantaje: nu trebuie sa standardizam datele(sa le aducem
in intervale de 0,1 cum am facut mai sus).
https://www.geeksforgeeks.org/random-forest-algorithm-in-machine-learning/
"""

model1=RandomForestClassifier()

model1.fit(X,y)

print(model1.score(X,y))#acuratete de 100% trebuie tratat cat mai critic

#vom imparti x si y in teste de antrenament si testare.

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15)#o sa sparga setul de date in 4 variabile
#separate destinate antrenarii si testatrii
#scopul este de a antrena modelul pe teste pe care nu le-a vazut,adica restul testelor 

model1=RandomForestClassifier()

model1.fit(X_train,y_train)#il antrenam pe datele de antrenament

print(model1.score(X_test,y_test))#procent de 80%

gender = input("Enter Gender (Male=2, Female=1): ")
applicant_income = input("Enter Applicant Income: ")

input_data = pd.DataFrame({
    'GenderEncoded': [gender],
    'ApplicantIncome': [applicant_income],
    'CoapplicantIncome':1000,
    'Credit_History':1000,
    'Dependents':1000,
    'Education':1,
    'LoannAmount':100
})

print(prediction=model1.predict(input_data));


