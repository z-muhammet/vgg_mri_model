# BrainMRIModel

Bu proje, beyin MRI görüntülerinden tümör tespiti ve sınıflandırması için derin öğrenme tabanlı bir model geliştirmeyi amaçlamaktadır. Proje, PyTorch çerçevesini kullanarak özel bir Evrişimsel Sinir Ağı (CNN) mimarisi eğitir ve model performansını artırmak için çeşitli gelişmiş eğitim stratejileri ve veri artırma teknikleri kullanır.

## Kullanılan Teknolojiler ve Kütüphaneler

*   **Derin Öğrenme Çerçevesi:** PyTorch
*   **Veri İşleme:** NumPy, PIL (Python Imaging Library), Albumentations
*   **Görselleştirme:** Matplotlib
*   **Optimizasyon:** `torch.optim`
*   **Geliştirme Dili:** Python

## Model Mimarisi (models/basic_cnn_model.py)

Projenin kalbinde, özel olarak tasarlanmış `BasicCNNModel` adında VGG benzeri bir Evrişimsel Sinir Ağı (CNN) bulunmaktadır.

*   **Temel Katmanlar:** `nn.Conv2d`, `nn.GroupNorm`, `nn.GELU`, `nn.Identity`, `nn.Dropout2d`, `nn.AdaptiveAvgPool2d`, `nn.BatchNorm1d`, `nn.Mish`, `nn.Linear`.
*   **`TrainOnlyNoise`:** Sadece eğitim sırasında girdi verisine gürültü ekleyen özel bir katman.
*   **`ConvBlock`:** Her biri evrişim, grup normalizasyonu ve aktivasyon içeren temel yapı taşları. Atlamalı bağlantılar (shortcut connections) içerir.
*   **Blok Yapısı:** Model, ayrı ayrı dondurulabilen ve açılabilen üç ana bloktan oluşur. Bu, aşamalı öğrenme stratejisini mümkün kılar.
*   **Dinamik Adaptörler:** Son açılan bloğun çıktısını sınıflandırma katmanına uygun hale getiren adaptör katmanları.
*   **Ağırlık Başlatma:** Kaiming Normal (He normal) ve sabit başlatma teknikleri kullanılmıştır.
*   **Aşamalı Öğrenme (Progressive Learning):** `freeze_blocks_until` metodu ile modelin katmanları kademeli olarak eğitilir. Başlangıçta yalnızca ilk blok (Blok 1) eğitilebilir durumdadır. Eğitim sırasında, belirli koşullar altında (şu anki uygulamada Blok 2, 15. epokta açılmaya zorlanır; Blok 3'ün açılması ise engellenir) diğer bloklar dondurulmuş durumdan çıkarılır.

    *   **Blok Detayları:**
        *   **Blok 1:** Giriş katmanlarını (örn. 3 kanal) alır ve 64 çıkış kanalı üretir. İki adet `ConvBlock` katmanı ve bir `Dropout2d` katmanı içerir. İkinci `ConvBlock` downsampling (`stride=2`) yapar.
        *   **Blok 2:** 64 giriş kanalı ile başlar ve 512 çıkış kanalı üretir. Üç adet `ConvBlock` katmanı ve bir `Dropout2d` katmanı içerir. Son `ConvBlock` downsampling (`stride=2`) yapar ve `dilation=2` kullanır.
        *   **Blok 3:** 512 giriş kanalı ile başlar ve 256 çıkış kanalı üretir. Üç adet `ConvBlock` katmanı ve bir `Dropout2d` katmanı içerir. Son `ConvBlock` downsampling (`stride=2`) yapar.
*   **Çıkış Dense Katmanları:**
    *   **Blok 1 Çıkışı:** 64 kanal çıkışı, 128 nöronlu bir dense katmanına bağlanır.
    *   **Blok 2 Çıkışı:** 512 kanal çıkışı doğrudan son dense katmanına bağlanır.
    *   **Blok 3 Çıkışı:** 256 kanal çıkışı, 128 nöronlu bir dense katmanına bağlanır.
    *   **Son Dense Katmanı:** Tüm blokların dense katmanları birleştirilerek 512 nöronlu bir ara katmana bağlanır ve son olarak 3 sınıflı çıkış katmanına (softmax) bağlanır.

## Blok Kullanımı ve Katman Sayısı

Modelin eğitimi, hangi blokta sonlandığına bağlı olarak farklı sayıda katman kullanır:

*   **Blok 1 ile Sonlanırsa:**
    *   2 adet `ConvBlock` (her biri 4 katman: Conv2d, GroupNorm, SiLU, Dropout2d)
    *   1 adet Dropout2d (0.1)
    *   1 adet AdaptiveAvgPool2d
    *   1 adet BatchNorm1d
    *   1 adet Linear (64->512) adaptör
    *   1 adet Linear (512->3) sınıflandırıcı
    *   Toplam: 2 adet conv2d - 2 adet dense

*   **Blok 2 ile Sonlanırsa:**
    *   Blok 1'in tüm katmanları
    *   3 adet `ConvBlock` (her biri 4 katman: Conv2d, GroupNorm, SiLU, Dropout2d)
    *   1 adet Dropout2d (0.2)
    *   1 adet BatchNorm1d
    *   1 adet Identity adaptör (512->512)
    *   1 adet Linear (512->3) sınıflandırıcı
    *   Toplam: ~ 2 + 2 adet conv2d 1 dense 

*   **Blok 3 ile Sonlanırsa: (Açılması şuanki Tasarım ile izin verilmiyor)**
    *   Blok 1 ve 2'nin tüm katmanları
    *   3 adet `ConvBlock` (her biri 4 katman: Conv2d, GroupNorm, SiLU, Dropout2d)
    *   1 adet Dropout2d (0.3)
    *   1 adet AdaptiveAvgPool2d
    *   1 adet BatchNorm1d
    *   1 adet Linear (256->512) adaptör
    *   1 adet Linear (512->3) sınıflandırıcı
    *   Toplam: ~2 + 2 + 3 adet conv2d 1 adet pooling 2 adet dense

Not: Her `ConvBlock` içindeki shortcut connection'lar da hesaba katıldığında, gerçek katman sayısı biraz daha yüksek olabilir. Ayrıca, eğitim sırasında bazı katmanlar dondurulmuş (frozen) olabilir, bu durumda aktif katman sayısı azalır.

## Eğitim Prosedürleri ve Algoritmaları (models/train.py)

Modelin eğitimi için çeşitli modern teknikler ve algoritmalar kullanılmıştır:

*   **Optimizasyon:** `torch.optim.AdamW`.
*   **Kayıp Fonksiyonu:** `nn.CrossEntropyLoss` (sınıf dengesizliğini gidermek için sınıf ağırlıkları ile).
*   **Öğrenme Oranı Çizelgeleyiciler (LR Schedulers):**
    *   **Başlangıç LR:** `1e-2` ile başlayan ve aşamalı olarak değişen öğrenme oranı stratejisi.
    *   **Isınma Fazı (Warmup):** İlk 5 epokta lineer artış ile `1e-2`'ye ulaşır.
    *   **Ana Eğitim Fazı:**
        *   `OneCycleLR`: 5-20. epoklar arasında maksimum LR'ye (`1e-2`) ulaşır ve sonra azalır.
        *   `CosineAnnealingLR`: 20-40. epoklar arasında kosinüs dalgası şeklinde LR'yi azaltır.
        *   `ReduceLROnPlateau`: 40. epoktan sonra doğrulama kaybı plato yaptığında LR'yi 0.1 faktörü ile azaltır.
    *   **LR Kısıtlamaları:**
        *   40-50. epoklar arasında minimum LR `1e-3`'ün altına düşmez.
        *   50. epoktan sonra minimum LR `1e-4`'ün altına düşmez.
        *   **Dinamik LR Ayarlaması:** Her epok sonunda doğrulama metriklerine göre LR'yi otomatik olarak ayarlar.
*   **Otomatik Karışık Hassasiyet (AMP):** `torch.amp.autocast` ve `torch.cuda.amp.GradScaler` ile eğitim hızı ve bellek verimliliği artırılır.
*   **Gradiyent Biriktirme (Gradient Accumulation):** Efektif batch boyutunu artırmak için gradyanlar birden fazla adımda biriktirilir.
*   **Rastgele Ağırlık Ortalaması (Stochastic Weight Averaging - SWA):** `AveragedModel` kullanılarak eğitimin son aşamalarında model ağırlıklarının ortalaması alınarak genelleme yeteneği artırılır. (Yapılandırmaya göre etkinleştirilebilir/devre dışı bırakılabilir.)
*   **Mixup:** Veri artırma tekniği olarak Mixup kullanılır. Girdi görüntüleri ve etiketleri lineer olarak karıştırılır. Mixup alfa değeri eğitimin erken aşamalarında lineer olarak azaltılır (0. epoktan 15. epoka kadar).
*   **Dinamik Düzenlileştirme:** Modelin aşırı veya az öğrenmesine göre dropout oranlarını dinamik olarak ayarlayan bir mekanizma.
*   **Sıcaklık Ölçeklendirme (Temperature Scaling):** Modelin tahmin güvenilirliğini kalibre etmek için kullanılır.
*   **Geri Alma Mekanizması (Rollback):** Performans düşüşlerinde modelin önceki iyi ağırlıklara dönmesini sağlar.
*   **Sınıf Dengeli Ağırlıklar:** Veri setindeki sınıf dengesizliğini ele almak için `get_class_balanced_weights` ile ağırlıklar hesaplanır ve `WeightedRandomSampler` ile DataLoader dengelenir.
*   **Sınıf Cezası (ClassPenalty):** Eğitim ve doğrulama doğruluğu arasındaki farka göre belirli sınıfların ağırlıklarını dinamik olarak ayarlar.
*   **Erken Durdurma:** Belirli bir epok boyunca performans artışı olmazsa eğitimi sonlandırır.
*   **Dinamik Veri Artırma:** Epoklara göre değişen şiddetlerde veri artırma stratejileri uygulanır.

## GPU Performans Optimizasyonları

Proje, GPU performansını en üst düzeye çıkarmak ve eğitim sürecini hızlandırmak için çeşitli modern teknolojiler ve teknikler kullanır:

*   **Otomatik Karışık Hassasiyet (Automatic Mixed Precision - AMP):**
    *   `torch.amp.autocast` ve `torch.cuda.amp.GradScaler` kullanarak hem hız hem de bellek verimliliği için Float32 ve Float16 veri tiplerini dinamik olarak karıştırır. Uyumlu GPU'larda (Tensor Core gibi) daha hızlı matematik işlemleri sağlar ve bellek kullanımını optimize eder.

*   **Gradiyent Biriktirme (Gradient Accumulation):**
    *   `GRADIENT_ACCUMULATION_STEPS` ile belirtilen adımlarla gradyanları biriktirerek daha büyük bir "efektif batch boyutu" simüle eder. Bu, GPU belleği kısıtlıyken bile büyük batch boyutlarının faydalarından yararlanmayı sağlar.

*   **`DataLoader` Optimizasyonları:**
    *   **`num_workers`:** Veri yüklemesini ayrı alt işlemlere dağıtarak CPU'nun veri hazırlığına, GPU'nun ise model eğitimine odaklanabilmesini sağlar. (Windows için kısıtlamalar nedeniyle otomatik olarak 0'a ayarlanır).
    *   **`pin_memory=True`:** Verinin GPU'ya kopyalanmasını hızlandırmak için CPU belleğinde sabitlenmiş bir alana yüklenmesini sağlar.
    *   **`persistent_workers=True`:** `DataLoader` alt işlemlerinin epochlar arasında yeniden başlatılmasını engelleyerek ek yükü azaltır ve veri yükleme süresini kısaltır. (Windows için kısıtlamalar nedeniyle devre dışı bırakılabilir).

*   **GPU Bellek Yönetimi:**
    *   **`torch.cuda.empty_cache()`:** GPU üzerindeki önbelleği boşaltarak bellek fragmentasyonunu ve gereksiz bellek kullanımını azaltır.
    *   **`gc.collect()`:** Python'ın çöp toplayıcısını manuel olarak tetikleyerek GPU belleğinin daha verimli serbest bırakılmasına yardımcı olur.
    *   **`set_to_none=True` ile `optim.zero_grad()`:** Gradyan tensörlerini doğrudan sıfırlamak yerine `None` olarak ayarlayarak bellek tahsisini optimize eder.

*   **`ConvBlock` İyileştirmeleri (Dolaylı Etki):**
    *   **`GroupNorm`:** Daha küçük batch boyutlarında stabilite sağlar ve bazı GPU mimarilerinde performansı artırabilir.
    *   **`kernel_size=3`:** Daha az hesaplama gerektirir ve daha derin ağların daha verimli çalışmasına olanak tanır.
    *   **`SiLU()` aktivasyon fonksiyonu:** Diğer bazı aktivasyonlara göre daha düşük hesaplama maliyetine sahip olabilir.
    *   **`nn.Dropout2d` (Spatial Dropout):** Uzamsal bağıntıları koruyarak daha etkili bir düzenlileştirme ve genelleme sağlar, bu da daha az epokta iyi sonuçlar elde etmeye yardımcı olabilir.

## Veri Yükleme ve Ön İşleme (dataset/custom_dataset.py)

Veri seti işlemleri için özel bir PyTorch `Dataset` sınıfı kullanılmıştır:

*   **`CustomTumorDataset`:** `.npy` formatındaki ön işlenmiş görüntülerden veri yükler.
*   **Önbellekleme:** Sık erişilen veriler için bellek önbelleklemesi kullanır.
*   **Görüntü Artırma ve Dönüşümler:** `Albumentations` kütüphanesi ile çeşitli rastgele dönüşümler, yeniden boyutlandırma ve normalizasyon uygulanır.

## Yardımcı Betikler

*   **`test.py`:** Eğitilmiş modelin test veri seti üzerindeki performansını değerlendirmek için kullanılır. Hem en iyi modeli hem de SWA modelini test edebilir.
*   **`losses.py`:** Gelecekte özel kayıp fonksiyonları eklemek için yer tutucu.
*   **Renk Kodlu Konsol Çıktısı:** Terminal çıktılarını renklendirmek için özel ANSI kaçış kodları kullanılır.
*   **Windows için Çoklu İşlem Desteği:** `multiprocessing` modülü, Windows işletim sistemlerinde uyumluluğu sağlamak için yapılandırılmıştır.

## Başlangıç

Projeyi çalıştırmak için gerekli bağımlılıkları yüklemeniz ve ardından `models/train.py` betiğini çalıştırmanız yeterlidir.


# Bağımlılıkları yükleyin (varsayılan olarak bir requirements.txt dosyasının olduğunu varsayalım)
pip install -r requirements.txt

# Modeli eğitin
python models/train.py

# Modeli test edin (eğitimden sonra)
python test.py
```

---

