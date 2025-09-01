 ###########################################
## Rating Product & Sorting Reviews in Amazon
###########################################

# Ürünün kalitesi ürünü aldırsa da ürünü aldıran önemli etkenlerden biri olarak kalabalığın bilgeliği kavramı olduğunu gördük.
# İş problemimizde hem ürüne verilen yorumları doğru bir şekilde sıralamak hem de ürüne verilen puanları dengeli bir şekilde
# hesaplamak olarak bahsedilmekte. Hedefimiz kullanıcıya sorunsuz bir satın alma yolculuğu yaşatmak.

###################################################
# İş Problemi
###################################################

# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış sonrası verilen puanların doğru şekilde hesaplanmasıdır.
# Bu problemin çözümü e-ticaret sitesi için daha fazla müşteri memnuniyeti sağlamak, satıcılar için ürünün öne çıkması ve satın
# alanlar için sorunsuz bir alışveriş deneyimi demektir. Bir diğer problem ise ürünlere verilen yorumların doğru bir şekilde sıralanması
# olarak karşımıza çıkmaktadır. Yanıltıcı yorumların öne çıkması ürünün satışını doğrudan etkileyeceğinden dolayı hem maddi kayıp
# hem de müşteri kaybına neden olacaktır. Bu 2 temel problemin çözümünde e-ticaret sitesi ve satıcılar satışlarını arttırırken müşteriler
# ise satın alma yolculuğunu sorunsuz olarak tamamlayacaktır.

###################################################
# Veri Seti Hikayesi
###################################################

# Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeşitli metadataları içermektedir.
# Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı puanları ve yorumları vardır.

# Değişkenler:
## reviewerID – Kullanıcı ID’si
## asin – Ürün ID’si.
## reviewerName – Kullanıcı Adı
## helpful – Faydalı değerlendirme derecesi
## reviewText – Değerlendirme
## overall – Ürün rating’i
## summary – Değerlendirme özeti
## unixReviewTime – Değerlendirme zamanı
## reviewTime – Değerlendirme zamanı
## day_diff – Değerlendirmeden itibaren geçen gün sayısı
## helpful_yes – Değerlendirmenin faydalı bulunma sayısı
## total_vote – Değerlendirmeye verilen oy sayısı

import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 10)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

###################################################
# GÖREV 1: Average Rating'i Güncel Yorumlara Göre Hesaplayınız ve Var Olan Average Rating ile Kıyaslayınız.
###################################################

# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır.
# Bu görevde amacımız verilen puanları tarihe göre ağırlıklandırarak değerlendirmek.
# İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.

###################################################
# Adım 1: Veri Setini Okutunuz ve Ürünün Ortalama Puanını Hesaplayınız.
###################################################

df = pd.read_csv("amazon_review.csv")
df.head()
df.shape ## (4915, 12)

## tüm veri setinin average ratıng'i
df["overall"].mean()
## 4.587589013224822
df.info()

###################################################
# Adım 2: Tarihe Göre Ağırlıklı Puan Ortalamasını Hesaplayınız.
###################################################
# kullanıcı trendinin etkisini barındıracak şekilde tarihe göre ağırlıklandırma yapacağız.
# buradan bir rating hesaplayacağız. kullanıcıların vermiş olduğu overall ortalamalarına bakarak, ağırlıklı ortalamalar
# ile kıyaslıyor olacağız.

# Tarihlere göre ağırlıklı puan hesabı yapabilmek için:
#   - reviewTime değişkenini tarih değişkeni olarak tanıtmanız
#   - reviewTime'ın max değerini current_date olarak kabul etmeniz
#   - her bir puan-yorum tarihi ile current_date'in farkını gün cinsinden ifade ederek yeni değişken oluşturmanız
#   - ve gün cinsinden ifade edilen değişkeni quantile fonksiyonu ile 4'e bölüp (3 çeyrek verilirse 4 parça çıkar)
#   - çeyrekliklerden gelen değerlere göre ağırlıklandırma yapmanız gerekir.
#   - örneğin q1 = 12 ise ağırlıklandırırken 12 günden az süre önce yapılan yorumların ortalamasını alıp bunlara
#   - yüksek ağırlık vermek gibi.
# day_diff: yorum sonrası ne kadar gün geçmiş

df['reviewTime'] = pd.to_datetime(df['reviewTime'], dayfirst=True)  #dayfirst=True, özellikle Avrupa ve Türkiye gibi ülkelerdeki tarih formatlarında (gün/ay/yıl) doğru tarih dönüşümü yapmak için kullanılır.
current_date = pd.to_datetime(str(df['reviewTime'].max()))  #max ile en son işlem yapılan tarihi current_date olarak alıyoruz
df["day_diff"] = (current_date - df['reviewTime']).dt.days


# day_diff değişkeninin çeyrek değerlerine göre ağırlıklı puanı hesaplama
df.loc[df["day_diff"] <= df["day_diff"].quantile(0.25), "overall"].mean() * 28 / 100 + \
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.25)) & (df["day_diff"] <= df["day_diff"].quantile(0.50)), "overall"].mean() * 26 / 100 + \
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.50)) & (df["day_diff"] <= df["day_diff"].quantile(0.75)), "overall"].mean() * 24 / 100 + \
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.75)), "overall"].mean() * 22 / 100
# 4.595593165128118
#bana en yakın olan tarihten en uzak olan tarihe doğru ağırlandırma yapılarak ortalama hesabı yapıldı.

# zaman bazlı ortalama ağırlıkların belirlenmesi
def time_based_weighted_average(dataframe, w1=24, w2=22, w3=20, w4=18, w5=16):
    return dataframe.loc[dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.2), "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.2)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.4)), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.4)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.6)), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.6)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.8)), "overall"].mean() * w4 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.8)), "overall"].mean() * w5 / 100

time_based_weighted_average(df)
#sonuçlar cok yakın pek risk alınmamış ve normal dağılıma yakın bir veriyle çalıştıgımızı söylemektedir.
#Güncel zamana ağırlık verilerek yapılan ortalama daha yüksek. onda birlik fark büyük anlam ifade edebilir iş bilgisine bağlı olarak.
#Yorum: güncel yorumları dikkate almak ağırlıklı ortalama artış gösterebilir diyebiliriz. Pazar yerinin güncel durumunu gözettiğimde bu ortalama yavaş yavaş artıyor olmakta.
#zaman bazlı hesaplarken fonksiyon kısmında dilimi 4'ten 5'e çıkarmak hassasiyeti de arttırarak ortalamanın daha artmış görünmesini sağlamıştır. Belki de her ay için yapılacak kadar dilime bölünse ortalama daha da artabilir.Homojenlik artıyor gibi düşünülebilir.


# Adım 3: Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız.

### 4.600026902863648
### tüm veri setindeki ratınglerın ortalaması 4.587589013224822
### pazar yeri olarak guncel durumu yansıtan guncel yorumlara gore yapılan agırlıklı ortalama ıle elde ettıgımız
### ortalama ratıng deger, daha yuksektir. Güncel yorumları dıkkate almak sonucu olumlu etkılemıstır.


###################################################
# Görev 2: Ürün için Ürün Detay Sayfasında Görüntülenecek 20 Review'i Belirleyiniz.
###################################################

###################################################
# Adım 1. helpful_no Değişkenini Üretiniz
###################################################

# Not:
# total_vote bir yoruma verilen toplam up-down sayısıdır.
# up, helpful demektir.
# veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.

## veri setimizde helpful_yes ve total_vote değerlerımız var ancak helpful_no degerlerımız yok
## bu sebeple total_vote - helpful_yes yaparsak helpful_no degerlerımıze ulasırız.Bu da bir feature engineering işlemidir. Ham veriden yeni bir değişken üretiyoruz.

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

# sadece kullanacaklarımızı alıyoruz
df = df[["reviewerName", "overall", "summary", "helpful_yes", "helpful_no", "total_vote", "reviewTime"]]
df.head()

###################################################
# Adım 2. score_pos_neg_diff, score_average_rating ve wilson_lower_bound Skorlarını Hesaplayıp Veriye Ekleyiniz
###################################################
# yoruma ve şansa yer bırakmadan, güvenebileceğimiz istatistiki bir metrik ile sıralama gerçekleştiremez miyiz?
# sorusuna cevap olarak : Wilson-loverband methodu ortaya çıkıyor.
# ilgili yoruma gelen pozitif ve negatif adetlere bakıp, bir güven aralığı oluşturan ve bu güven aralığı içerisindeki
# minimum değeri hesaplayan bir değer oluşturuyor. elimizdeki değer için bir alt ve üst aralık oluşturuyoruz.
# artık yeni gelen kişi %95 güvenle bu değer arasında yer alır diyebiliyoruz. güven aralığı sayesinde hesaplanan minimum
# değeri alarak, aslında en kötü durumu değerlendirerek, durumlarımızı bu sıralamaya göre şansa yer bırakmadan
# gerçekleştirebilirim diyor. Bernoulli dağılımı kullanarak içerisinde bir hesaplama gerçekleştiriliyor. bir rasgele
# değişkeni olası iki sonucu olduğunda, kullanılan bir olasılık hesabı oluyor.
# fonksıyonda up, down yorumlarını verıyoruz ve guven aralıgını tanımlıyoruz.

#bernoulli dağılımı 1 ve 0 lardan oluşan bir dağılım çeşidi. bunu kullanıyor olmak bu fonku istatistiki temellere dayandırıyor olduğumu gösterir.

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)  #yüzde kaç güvenle z-skoru gerekir onu bulur.
    phat = 1.0 * up / n  #Toplam oy içindeki pozitif oran.
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)
        #Wilson'un formülü şunu yapar:
        # Gerçek pozitif oranı bilmiyoruz ama bu oran güven aralığı içinde hangi noktadan aşağı düşmez, onu hesaplıyoruz.
        # Bu değer, yukarıdan değil, aşağıdan sınır koyar → “emin olduğumuz minimum başarı oranı”.

        #WLB

def score_up_down_diff(up, down):
    return up - down
#bu yöntemde ise sadece bir fark alma mevcut bir range elde edilir.


def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)
# bu fonkda ise de oransal bir hesap yapmış oluyruz.
# dezavantajı ise; pozitif ve negatif oylardaki durum etkisini burada da sürdürüyor.
# yani scoreupdown ve scoreaverageratingde topluluğun bilgeliği etkisi burada gözlemlenmiyor.

# score_pos_neg_diff
df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

# score_average_rating
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)
df.sort_values("score_average_rating",ascending=False).head(20)
# sıraladığımızda score average ratıngı 1 olanlar yukarıda gelır. social proof etkisi burada gozlenlenmez.
# yanı kısılerın verdıgı helpful_yes degerlerıne gore, score average ratıng yüksek geliyor. ama kalabalıgın bılgelıgı kavramı burada bulunmuyor.

# wilson_lower_bound
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)
    # WLB hem orana hemde oy sayısına bakar ve %95 güvenle "en az bu kadar iyidir" diyerek güvenilir, dengeli bir sıralama üretir.
##################################################
# Adım 3. 20 Yorumu Belirleyiniz ve Sonuçları Yorumlayınız.
###################################################

df.sort_values("wilson_lower_bound", ascending=False).head(20)

# ilk satıra baktıgımızda kısının wılson lower boundu 0.95 gelmiş. total votes 2020, helpful yes 1952, helpful no 68,
# 2020 toplam oy, kalabalığın bilgeliğini yansıtıyor. buraya oldukca fazla kısı yorum yapmıs ve kalabalıgın bılgelıgı dıkkate alınmıs
# average ratınge gore sıralasaydık ve 0.91 average ratınge sahıp satıra baksaydık, burada total vote a baktıgımızda
# sadece 49 tane yorum aldıgını gormekteyız. kısı sayısı az oldugundan genellebılırlık kaygısı var. topluluğun bilgeliği kavramı yok.

# bu yorum %95 guvenilirlikle minimum %70 oraninda helpfulyes alir(WLB'de)


"""
                              reviewerName  overall                                            summary  helpful_yes  helpful_no  total_vote reviewTime  score_pos_neg_diff  score_average_rating  wilson_lower_bound
2031                  Hyoun Kim "Faluzure"  5.00000  UPDATED - Great w/ Galaxy S4 & Galaxy Tab 4 10...         1952          68        2020 2013-01-05                1884               0.96634             0.95754  ## bu yorum %95 guvenılırlıkle benım sectıgım kıtlede mın %95 oranında helpfull_yes olacak.
3449                     NLee the Engineer  5.00000  Top of the class among all (budget-priced) mic...         1428          77        1505 2012-09-26                1351               0.94884             0.93652
4212                           SkincareCEO  1.00000  1 Star reviews - Micro SDXC card unmounts itse...         1568         126        1694 2013-05-08                1442               0.92562             0.91214
317                Amazon Customer "Kelly"  1.00000                                Warning, read this!          422          73         495 2012-02-09                 349               0.85253             0.81858
4672                               Twister  5.00000  Super high capacity!!!  Excellent price (on Am...           45           4          49 2014-07-03                  41               0.91837             0.80811
1835                           goconfigure  5.00000                                           I own it           60           8          68 2014-02-28                  52               0.88235             0.78465
3981            R. Sutton, Jr. "RWSynergy"  5.00000  Resolving confusion between "Mobile Ultra" and...          112          27         139 2012-10-22                  85               0.80576             0.73214
3807                            R. Heisler  3.00000   Good buy for the money but wait, I had an issue!           22           3          25 2013-02-27                  19               0.88000             0.70044
4306                         Stellar Eller  5.00000                                      Awesome Card!           51          14          65 2012-09-06                  37               0.78462             0.67033
4596           Tom Henriksen "Doggy Diner"  1.00000     Designed incompatibility/Don't support SanDisk           82          27         109 2012-09-22                  55               0.75229             0.66359
315             Amazon Customer "johncrea"  5.00000  Samsung Galaxy Tab2 works with this card if re...           38          10          48 2012-08-13                  28               0.79167             0.65741
1465                              D. Stein  4.00000                                           Finally.            7           0           7 2014-04-14                   7               1.00000             0.64567
1609                                Eskimo  5.00000                  Bet you wish you had one of these            7           0           7 2014-03-26                   7               1.00000             0.64567
4302                             Stayeraug  5.00000                        Perfect with GoPro Black 3+           14           2          16 2014-03-21                  12               0.87500             0.63977
4072                           sb21 "sb21"  5.00000               Used for my Samsung Galaxy Tab 2 7.0            6           0           6 2012-11-09                   6               1.00000             0.60967
1072                        Crysis Complex  5.00000               Works wonders for the Galaxy Note 2!            5           0           5 2012-05-10                   5               1.00000             0.56552
2583                               J. Wong  5.00000                  Works Great with a GoPro 3 Black!            5           0           5 2013-08-06                   5               1.00000             0.56552
121                                 A. Lee  5.00000                     ready for use on the Galaxy S3            5           0           5 2012-05-09                   5               1.00000             0.56552
1142  Daniel Pham(Danpham_X @ yahoo.  com)  5.00000                          Great large capacity card            5           0           5 2014-02-04                   5               1.00000             0.56552
1753                             G. Becker  5.00000                    Use Nothing Other Than the Best            5           0           5 2012-10-22                   5               1.00000             0.56552
"""
