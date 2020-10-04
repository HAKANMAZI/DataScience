# optimizer : sonuca ne kadar yakın olduğumuz hesaplayan algoritma çeşitleri
from.keras.optimizer import Adam

# Feature exraction 
model.add(Conv2D(20, (5,5), padding="same", input_shape=5))
  # 20 : filters : the dimensionality of the output space
  # (5,5) : kernel_size : specifying the length of the 1D convolution window
  # (5,5) : görüntüyü 5 e 5 lik görüntü ile çarpar
  # padding : One of "valid", "causal" or "same" (case-insensitive). "valid" means "no padding". "same" results in padding the input such that the output has the same length as the original input. "causal" results in causal (dilated) convolutions,
model.add(Activation("relu"))
  # Activation() : standartlaştırma
  # relu : Rectifier Lineer Unity, 0 ile 1 arasında değerler verir.
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
  # pool_size : 2 ye 2 aralığındaki maximum intensity değerlerini alarak resimde sürükletilir. yani 2 ye 2 arasında 4 değer var bunun en yüksek düğerlisini alır
  # strides : poolingde kaydıracağımız pencereyi nasıl kaydıracağımızı ayarlar. 2 ye 2 pencere olarak kaydırır.
  
# Klasik neurol network
model.add(Flatten())
  # Flatten() : Görüntüyü tek bir array'e çevirir. Düzleştirir. Tek boyuta çevirir.
model.add(Dense(500))
  # Dense(500) : 500 tane nöronluk Dense layer ekledik
model.add(Activation())

#softmax classifier, çıkış katmanı 
model.add(Dense(1))
  # 1 : çıkış katmanı 1 olur
model.add(Activation="softmax"))
  # softmax : ya 1 ya 0 verir.

model.compile(loss="binary_crossentrapy",
             optimizer=Adam()
             matrix = ["accuracy"])
  # loss : gerçek değer ile hatalı değer arasındaki hata oranını gösterir.

# start Training 
model.fit()
# modeli save etmemiz lazım çünkü sonrasında tekrar tekrar kullanacağız
model.save("hakan.model")

################################
model = load_model("hakan.model") 
(negative, pozitive) = model.predict(image)[0]

################################ Datai Team ################################
Not: ML tekniklerinden sadece bir tanesi DL, data'dan kendi kendine öğrenme
  Neden Deep Learning
    Çok fazla data var.
    Persformans açısından ML düşük data çok olunca
    1 milyondan fazla data 
    Big amound of Data
    Speech Recognition
    İmage Recognition
    Natural Language NLP
    Recommendation System
    ML covers DL

x_df = np.load('../x.npy')
x_df[26].reshape(64,64) # 26.indeksli np data arrayı 64*64 lük resim yapar. 
X = np.concatenate( (x_df[10:15], x_df[50:55]), axis=0) # 10dan 15e kadar ve 50den 55e kadar olan arrayleri axis=0 ile tek bir kolona ekler


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
