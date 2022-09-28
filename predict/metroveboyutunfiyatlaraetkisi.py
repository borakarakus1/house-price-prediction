# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 21:58:04 2021

@author: BoraKarakus
"""

# kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression

# Veriyi Yüklemek
data = pd.read_csv('emlak_fiyat_boyut_metro.csv')
    # verinin ilk 5 satırını kontrol edelim
#print(data.head())

# verinin şeklini kontrol etmek
#print(data.shape) # 100 satır 3 sutün

# veriyi tanıyalım
#print(data.describe()) # ortalama , standart sapma , min ,max değerleri görebiliriz.

# Regresyonu oluşturmak
    # bağımlı ve bağımsız değişkenleri tespit etmek ve bildirmek
    
# bağımsız değişkenlerimiz : boyut ve metroya yakınlık
x = data[['boyut','metro_yakinlik']]
# bağımlı değişken : fiyat
y = data['fiyat']

# Saçılım grafiğini çizelim (önce yatay ekseni ,sonra dikey ekseni yazıyoruz)
#plt.scatter(x['boyut'],y)
#eksenleri isimlendirmek
#plt.xlabel('Boyut',fontsize=15)
#plt.ylabel('Fiyat',fontsize=15)
#grafiği göster
#plt.show() # fiyat ile boyutun arasındaki pozitif yönde bir doğrusal ilişki olduğunu görebiliyoruz.

#plt.scatter(x['metro_yakinlik'],y)
#plt.xlabel("Metroya Yakınlık",fontsize=15)
#plt.ylabel("Fiyat",fontsize=15)
#plt.show() # metroya yakınlığın fiyat üzerinde negatif bir doğrusal ilişkisi olduğunu gördük yani metroya yakınlık azaldıkça evlerin fiyatı düşüyor

# Regresyon
# Doğrusal bi regresyon nesnesi oluşturarak başlıyoruz
reg = LinearRegression()
#Tüm öğrenme sürecinin özü burasıdır: regresyon uydurma yani fit etme
# ilk argümana bağımsız değişkeni ikincisine ise bağımlı değişkeni yazıyoruz.
reg.fit(x,y)

# Sabiti (Kesim noktasını) bulmak

# regresyonun kesme noktasını bulmak
# genellikle tek bir değer beklediğimizden sonucun float olacağını bilmeliyiz.
reg.intercept_

# Katsayıları Bulmak 
# regresyon katsayılarını elde etmek
# iki bağımsız değişken oldıuğu için 2 katsayı bekliyoruz.
reg.coef_

# R-Kareyi Hesaplamak
r = reg.score(x,y)


# Düzeltlmiş R kare hesaplamak
# düzeltilmiş r kare metriğinin oluşturulmasının kolaylaştırmak için x'in şeklini bilmeliyiz.
shape = x.shape
# düzeltilmiş r kareyi bulmak için , r kareyi , gözlem sayısını ve özellik sayısını bilmeliyiz.
r2 = reg.score(x,y)
# gözlem sayısı(n) , eksen 0 boyunca olan şekildir.
n = x.shape[0]
# özellik sayısı (öngörücüler,p) eksen 1 boyunca şeklidir.
p = x.shape[1]
# Düzeltilmiş r kareyi bulmak için kullanılan formul
duzeltilmis_r2 = 1 -(1-r2)*(n-1)/(n-p-1)


# Tahmin Yapma
# 120 metrekarelik ve metroya 10 dakika mesafelik bir dairenin fiyatının tahmini

predict = reg.predict([[120,10]])
predict2 = reg.predict([[150,3]])


# Tek Bir veri yerine bir veri çerçevesi tahmin etmek istersek veri çerçevesi oluşturalım
#yeni_data = pd.DataFrame({'boyut':[100,150,200],'metroya_yakinlik':[30,10,5]})

#♥ son olarak tahminleri veri çerçevesinde yeni bir kolonda saklayabiliriz.
#yeni_data['Tahmini Fiyat'] = reg.predict(yeni_data)
#print(yeni_data)

# Değişkenlerin p değerlerini hesaplamak
# feature selection modülünü içeri aktarmamız gerek
# bu modül regresyonumuz içim en uygun özellikleri seçmemizi sğlar
from sklearn.feature_selection import f_regression

# f_regression , bize boyut-fiyat , metroya yakınlık - fiyat regresyonların f istatiklerini hesaplıyor. Bu yaklaşım , iki özelliğin karşılıklı etkisini dikkate almıyor yani metroya yakınlık ve boyut arasındaki bağlantıyı dikkate almıyor
f_regression(x,y) # bize 2 çıktı verir. birinci regresyonlarını her biri için F istatiklerini içerir. İkincisi bu F istatistiklerinin p değerlerini içerir.
p_values = f_regression(x,y)[1]
#print(p_values.round(3)) # daha anlamlı olması için yuvarlıyoruz

# Bulgularımızı Tabloya Eklemek
reg_summary = pd.DataFrame(data=x.columns.values,columns=['Özellikler'])
reg_summary['Katsayilar'] = reg.coef_
reg_summary['p-degerleri'] = p_values.round(2)
print(reg_summary)