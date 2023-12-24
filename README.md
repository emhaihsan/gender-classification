# Gender Classification dengan Transfer Learning

## Latar Belakang
Pembuatan aplikasi klasifikasi foto berdasarkan gender menjadi penting dalam berbagai konteks penggunaan, seperti analisis demografis untuk survei sosial dan penelitian pasar, periklanan dan pemasaran yang efektif dengan menargetkan pasar sesuai gender, penyajian konten yang dipersonalisasi di platform media sosial, serta kontribusi terhadap keamanan publik dan penanganan kejahatan melalui identifikasi gender dari gambar atau rekaman video. Selain itu, aplikasi ini dapat mendukung pengembangan produk yang disesuaikan dengan preferensi gender. Meski memberikan manfaat, perlu diingat bahwa penggunaan data pribadi, termasuk identifikasi gender, harus mematuhi peraturan privasi yang ketat, dan aplikasi ini memerlukan pengujian hati-hati untuk mengurangi bias dan kesalahan klasifikasi yang tidak diinginkan serta mempertimbangkan masalah etika dan privasi.

## Tujuan dan Sasaran
Tujuan pembuatan aplikasi klasifikasi foto berdasarkan gender adalah untuk mengidentifikasi dan memisahkan foto-foto berdasarkan jenis kelamin (pria atau wanita) dari subjek dalam gambar. Hal ini dapat digunakan untuk berbagai keperluan, seperti analisis demografis, penargetan pasar, pengalaman pengguna yang dipersonalisasi, dan keamanan kejahatan. Aplikasi ini dapat membantu dalam mengumpulkan data demografis yang penting, menyajikan iklan yang lebih sesuai, menyediakan konten yang dipersonalisasi, mendukung penanganan kejahatan, dan mengembangkan produk yang sesuai dengan preferensi gender tertentu.

## Metodologi
### Data Understanding
CelebA (Celebrities Attributes) merupakan dataset yang berisi sejumlah wajah publik figur. Dataset ini sering digunakan untuk tugas-tugas di bidang computer vision, seperti deteksi wajah, pengenalan wajah, pengenalan emosi, analisis atribut wajah, dan lain sebagainya.

![CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA/intro.png)

Berikut adalah beberapa informasi umum tentang dataset CelebA:

* Jumlah data: Dataset CelebA terdiri dari lebih dari 200.000 gambar wajah selebriti.

* Anotasi: Setiap gambar dalam dataset ini dilengkapi dengan beberapa anotasi atribut, seperti jenis kelamin, kehadiran senyum, jenis rambut, dan banyak lagi. Anotasi ini digunakan untuk melatih model pembelajaran mesin untuk mengenali dan mengklasifikasikan atribut wajah dalam gambar.

* Format: Gambar-gambar dalam dataset ini biasanya berukuran beragam, tetapi umumnya berukuran $178\times218$ piksel. Format file umum yang digunakan adalah gambar JPEG.

Dataset ini banyak digunakan secara luas oleh para peneliti dan praktisi dalam komunitas computer vision untuk mengembangkan dan mengevaluasi berbagai model. Selain itu, juga terdapat tantangan tersendiri karena pada dataset ini terdapat banyak variasi dalam posisi wajah, ekspresi, dan cahaya, yang mana hal tersebut dapat membantu dalam pengembangan model yang tangguh dan tahan outlier.
### Data Preparation
Dari total 200 ribu data yang terdapat pada dataset CelebA, diberikan 5017 untuk diolah oleh tim Indonesia AI. Yang mana setelah ditelusuri terdapat duplikasi sebanyak 17 data sehingga total data yang digunakan adalah berjumlah 5000 gambar. Eksperimen ini tidak berfokus detail ke setiap langkah pada implementasi, melainkan hanya sebagai gambaran sekilas tentang penggunaan transfer learning. Sehingga tidak banyak dilakukan preproses dalam eksperimen ini, selain memastikan tidak ada data duplikat seperti yang sudah disebutkan di atas, eksplorasi data lainnya yang dilakukan adalah dengan melihat bar plot pada label laki-laki dan perempuan.

![gambar distribusi](https://github.com/mhihsan/gender-classification/blob/main/img/grafik.png)

Dapat dilihat bahwa data dengan label 0 (perempuan) memiliki jumlah data yang lebih banyak dibandingkan dengan data dengan label 1 (laki-laki).
### Pemodelan
Pada percobaan ini, dilakukan percobaan dengan melakukan transfer learning menggunakan 3 buah arsitektur *pretrained* di bawah ini:
#### 1. Inception V3:
Inception V3 adalah model jaringan saraf konvolusi yang dikembangkan oleh tim Google. Model ini menggunakan blok "Inception" yang kompleks untuk efisiensi komputasi dan mengatasi masalah vanishing gradients.

![Inception](https://production-media.paperswithcode.com/methods/inceptionv3onc--oview_vjAbOfw.png)

Referensi Paper:

*"Rethinking the Inception Architecture for Computer Vision"*
Authors: Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna

[Paper link](https://arxiv.org/abs/1512.00567v3)

#### 2. ResNet (Residual Network):
ResNet adalah model jaringan saraf konvolusi yang diusulkan oleh Kaiming He dan rekan-rekannya. Model ini menggunakan blok "residual" untuk mengatasi masalah gradien yang menghilang saat melatih jaringan yang sangat dalam.

![ResNet](https://miro.medium.com/v2/resize:fit:1400/format:webp/0*9LqUp7XyEx1QNc6A.png)

Referensi Paper:

*"Deep Residual Learning for Image Recognition"*
Authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun

[Paper link](https://arxiv.org/abs/1512.03385)

#### 3. Xception:
Xception adalah model jaringan saraf konvolusi yang mengembangkan ide dari Inception V3. Ini menggunakan "depthwise separable convolution" untuk menggantikan konvolusi tradisional, mengurangi jumlah parameter dan meningkatkan efisiensi.

![Xception](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*hOcAEj9QzqgBXcwUzmEvSg.png)

Referensi Paper:

*"Xception: Deep Learning with Depthwise Separable Convolutions"*

Authors: François Chollet

[Paper link](https://arxiv.org/abs/1610.02357)

Implementasi transfer learning menggunakan ketiga algoritma dapat dilihat pada link yang terdapat pada tabel di bawah ini:
| No      | Model       | Fine Tuning di Layer | Link |
|--- | --- | ---|---|
| 1   | InceptionV3   | 40 layers terakhir | https://github.com/mhihsan/gender-classification/blob/main/inception.ipynb |
| 2 | Resnet50V2 | 52 layer terakhir | https://github.com/mhihsan/gender-classification/blob/main/resnet.ipynb |
| 3 | Xception | 30 layer terakhir | https://github.com/mhihsan/gender-classification/blob/main/xception.ipynb |

Selain fine-tuning, ketiga notebook di atas dilatih menggunakan parameter yang identik. Parameter yang digunakan adalah sebagai berikut:
* Ukuran Gambar  : $218 \times 178$
* Pembagian Train-Val-Test  : $70-20-10$
* Pre-Trained Weight  : imagenet
* Fully Connected  : Pre Trained Feature -> Flatten -> Dense ($256$, relu) -> Dropout ($0.5$) -> Dense ($512$, relu) -> Dense ($2$, softmax)
* Optimizer  : Adaptive Momentum (Adam)
* Learning Rate  : $0.0001$
* Loss Function  : Categorical Crossentropy
* Jumlah Epoch : $20$

## Kinerja Model
Berikut hasil dari eksperimen yang dilakukan:

| No      | Model       | Val Accuracy | Val Loss | Test Accuracy | Test Loss
|--- | --- | --- |--- | --- | --- |
| 1   | InceptionV3   | $0.957$ | $0.190$ | $0.968$ | $0.155$ |
| 2 | Resnet50V2 | $0.932$ | $0.216$ | $0.934$ | $0.171$ |
| 3 | Xception | $0.929$ | $0.202$ |$0.952$ | $0.194$|

Dari eksperimen yang telah dilakukan, berdasarkan parameter yang telah disebutkan. Model transfer learning dengan InceptionV3 sebagai pre-trained mendapatkan hasil yang paling baik dibandingkan yang lain. Berikut grafik akurasi, loss, dan confusion matrix dari model InceptionV3:

![Grafik Loss](https://github.com/mhihsan/gender-classification/blob/main/img/loss.png)

![Grafik Akurasi](https://github.com/mhihsan/gender-classification/blob/main/img/akurasi.png)

![Confusion Matrix](https://github.com/mhihsan/gender-classification/blob/main/img/confusionmatrix.png)

![Prediksi Salah](https://github.com/mhihsan/gender-classification/blob/main/img/wrongprediction.png)

Bisa dilihat bahwa dari 20 epoch yang dijalankan, model sudah konvergen pada epoch-epoch awal. Hal ini juga terjadi pada penerapan model-model lain seperti Resnet dan Xception. Dari sini setidaknya bisa diasumsikan dua hal. Pertama ada kemungkinan bahwa penggunaan learning late $10 \times 10^{-4}$ sudah terlalu besar, karena model ini menggunakan pretrained weight dari imagenet. Kedua ada kemungkinan juga bahwa model sudah bekerja dengan baik dan maksimal, meskipun terlihat tidak meningkat dari segi loss, akurasi pada data tes sudah mencapai di atas 95%, sehingga bisa diasumsikan juga demikian.

Adapun jika kita melihat pada hasil gambar yang salah prediksi, ciri-ciri yang terlihat juga sekilas dapat mengecoh manusia seperti rambut panjang, bentuk wajah, dan penggunaan aksesoris. Dari sini kira-kira bisa diasumsikan bahwa kesalahan yang dilakukan oleh model mungkin masih bisa masuk ke dalam kategori wajar.

## Pengaplikasian sederhana dengan Streamlit 
Untuk mencoba apakah model bisa diimplementasikan ke dalam aplikasi. Dibuat sebuah web app sederhana dengan menggunakan [streamlit](https://streamlit.io/). Tampilan dari aplikasi yang dibuat adalah sebagai berikut:

![streamlit app](https://github.com/mhihsan/gender-classification/blob/main/img/streamlit.png)

Kode dari implementasi di atas bisa dilihat [di sini](https://github.com/mhihsan/gender-classification/blob/main/genderclf.py).





