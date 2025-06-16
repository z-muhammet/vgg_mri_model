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

*   **Temel Katmanlar:** `nn.Conv2d`, `nn.GroupNorm`, `nn.SiLU`, `nn.Identity`, `nn.Dropout2d`, `nn.AdaptiveAvgPool2d`, `nn.BatchNorm1d`, `nn.Mish`, `nn.Linear`.
*   **`TrainOnlyNoise`:** Sadece eğitim sırasında girdi verisine gürültü ekleyen özel bir katman.
*   **`ConvBlock`:** Her biri evrişim, 32 gruplu grup normalizasyonu ve `SiLU` aktivasyonu içeren temel yapı taşları. Bu bloklar, bir `Dropout2d` ve `TrainOnlyNoise` katmanı da içerir. Kalıntı bağlantılar (shortcut connections) bu blok içinde uygulanmamıştır.
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

*   **Optimizasyon:** `torch.optim.AdamW` optimizer, `5e-4` ağırlık azaltma (`weight_decay`) ile kullanılır. Bu, modelin aşırı öğrenmesini engellemeye yardımcı olur.
*   **Kayıp Fonksiyonu:** `nn.CrossEntropyLoss` kullanılır. Sınıf dengesizliğini gidermek amacıyla `get_class_balanced_weights` fonksiyonu tarafından hesaplanan sınıf ağırlıkları (`class_weights`) ile birlikte kullanılır. Bu ağırlıklar, her 10 epokta bir normalize edilir ve eğitim-doğrulama doğruluğu arasındaki farka göre dinamik olarak ayarlanır (ClassPenalty mekanizması).
*   **Öğrenme Oranı Çizelgeleyiciler (LR Schedulers):** Esnek ve dinamik bir öğrenme oranı stratejisi uygulanır. Başlangıç öğrenme oranı `1e-2`'dir.
    *   **`OneCycleLR`:** İlk `COSINE_START_EPOCH` (varsayılan 20) epoka kadar maksimum öğrenme oranına ulaşır ve sonra kosinüs anneal stratejisi ile azaltır. Bu, hızlı yakınsama ve daha iyi genelleme sağlar.
    *   **`CosineAnnealingLR`:** `COSINE_START_EPOCH`'tan `SWA_START_EPOCH - 1`'e kadar (varsayılan 20-60 arası) öğrenme oranını kosinüs dalgası şeklinde `1e-5` minimum değere kadar azaltır.
    *   **`LinearLR` (SWA Fazı için):** `SWA_START_EPOCH`'tan (`NUM_EPOCHS + 1`, yani varsayılan 61. epoktan itibaren) eğitim sonuna kadar sabit bir öğrenme oranı (`SWA_LR = 5e-4`) korur. Bu, SWA modelinin stabil bir şekilde yakınsamasına yardımcı olur.
    *   **`SequentialLR`:** Yukarıdaki çizelgeleyicileri belirli mihenk taşlarında (`milestones`) sırayla etkinleştirir.
    *   Alternatif olarak, doğrulama kaybı plato yaptığında öğrenme oranını `0.2` faktörü ile azaltan `ReduceLROnPlateau` çizelgeleyici (`patience=3`, `min_lr=1e-7`) de kullanılabilir.
*   **Mixup:** Girdi görüntüleri ve etiketleri lineer olarak karıştırılarak veri çeşitliliğini artıran bir veri artırma tekniğidir. `mixup_data` fonksiyonu tarafından `alpha=0.2` ile uygulanır. Mixup alfa değeri eğitimin erken aşamalarında (0. epoktan 15. epoka kadar) lineer olarak `0.0`'a düşürülür, bu da modelin başlangıçta daha fazla çeşitlilikle eğitilmesini, sonrasında ise gerçek verilere odaklanmasını sağlar.
*   **Dinamik Düzenlileştirme (`DynamicRegularization`):** Modelin eğitim performansına (aşırı öğrenme veya az öğrenme) göre dropout oranlarını dinamik olarak ayarlayan bir mekanizmadır. Aşırı öğrenme belirtileri (`tr_acc - val_acc > OVERFIT_THRESHOLD`) görüldüğünde dropout faktörünü artırır, az öğrenme (`val_acc < UNDERFIT_THRESHOLD`) durumunda ise azaltır.
*   **Sıcaklık Ölçeklendirme (`TemperatureScaler`):** Modelin tahmin güvenilirliklerini kalibre etmek için kullanılan öğrenilebilir bir sıcaklık parametresi (`temperature`) içerir. Eğitim sonunda SWA modeli üzerinde kalibre edilir.
*   **Geri Alma Mekanizması (Rollback):** Doğrulama doğruluğunda belirgin bir düşüş veya kayıpta artış olduğunda (`acc_drop >= ROLLBACK_ACC_THRESHOLD` veya `loss_rise >= ROLLBACK_LOSS_THRESHOLD`), modelin daha önceki en iyi ağırlıklarına geri dönmesini sağlar. Bu, istenmeyen performans düşüşlerinin önüne geçmek için önemli bir stratejidir. Maksimum 10 geri alma işlemi (`MAX_TOTAL_ROLLBACKS`) ve bir soğuma süresi (`ROLLBACK_COOLDOWN = 5` epok) içerir.
*   **Sınıf Cezası (ClassPenalty):** Eğitim ve doğrulama doğruluğu arasındaki sınıf bazlı fark (`per_class_gap`) belirli eşikler (`PENALTY_GAP_LOW`, `PENALTY_GAP_HIGH`) içinde olduğunda ve 30. epokdan sonra etkinleşir. Bu durumda, daha kötü performans gösteren sınıfların ağırlıklarını artırarak (`1.03` faktörle) ve daha iyi performans gösteren sınıfların ağırlıklarını azaltarak (`0.97` faktörle) sınıf dengesizliğini eğitim süresince dinamik olarak düzenler. Ayrıca, 40, 50 ve 55. epoklarda '1_menin' sınıfı için özel ağırlık artışları uygulanır.
*   **Erken Durdurma:** `EARLY_STOPPING_PATIENCE` (varsayılan 18) epok boyunca doğrulama doğruluğunda iyileşme olmazsa eğitim durdurulur. En iyi model ağırlıkları bu süre zarfında kaydedilir.
*   **Rastgele Ağırlık Ortalaması (Stochastic Weight Averaging - SWA):** `SWA_START_EPOCH`'tan itibaren (`NUM_EPOCHS + 1`, yani varsayılan 61. epok) `AveragedModel` kullanılarak model ağırlıklarının ortalaması alınarak genelleme yeteneği artırılır. Eğitim sonunda `update_bn` ile BatchNorm katmanları güncellenir ve SWA modeli ayrı bir checkpoint olarak kaydedilir.
*   **Aşamalı Öğrenme (Progressive Learning):** `models/basic_cnn_model.py`'de tanımlanan `freeze_blocks_until` metodu ile modelin katmanları kademeli olarak eğitilir. Eğitim sırasında, belirli koşullar altında (şu anki uygulamada Blok 2, 15. epokta açılmaya zorlanır; Blok 3'ün açılması ise engellenir) diğer bloklar dondurulmuş durumdan çıkarılır.

## GPU Performans Optimizasyonları

Proje, GPU performansını en üst düzeye çıkarmak ve eğitim sürecini hızlandırmak için çeşitli modern teknolojiler ve teknikler kullanır:

*   **Otomatik Karışık Hassasiyet (Automatic Mixed Precision - AMP):** `torch.amp.autocast` ve `torch.cuda.amp.GradScaler` kullanarak hem hız hem de bellek verimliliği için Float32 ve Float16 veri tiplerini dinamik olarak karıştırır. Uyumlu GPU'larda (Tensor Core gibi) daha hızlı matematik işlemleri sağlar ve bellek kullanımını optimize eder. `amp_enabled` bayrağı ile kontrol edilir ve yalnızca CUDA cihazı mevcut ve BF16 destekliyorsa etkinleştirilir.
*   **Gradiyent Biriktirme (Gradient Accumulation):** `GRADIENT_ACCUMULATION_STEPS` (varsayılan 4) ile belirtilen adımlarla gradyanları biriktirerek daha büyük bir "efektif batch boyutu" simüle eder. Bu, GPU belleği kısıtlıyken bile büyük batch boyutlarının faydalarından yararlanmayı sağlar. `loss / GRADIENT_ACCUMULATION_STEPS).backward()` ile uygulanır.
*   **`DataLoader` Optimizasyonları:**
    *   **`num_workers`:** Veri yüklemesini ayrı alt işlemlere dağıtarak CPU'nun veri hazırlığına, GPU'nun ise model eğitimine odaklanabilmesini sağlar. Windows işletim sistemleri için `0` olarak ayarlanır ve `persistent_workers` devre dışı bırakılır (`freeze_support_for_win` fonksiyonu ile).
    *   **`pin_memory=True`:** Verinin GPU'ya kopyalanmasını hızlandırmak için CPU belleğinde sabitlenmiş bir alana yüklenmesini sağlar.
    *   **`persistent_workers=True`:** `DataLoader` alt işlemlerinin epochlar arasında yeniden başlatılmasını engelleyerek ek yükü azaltır ve veri yükleme süresini kısaltır. (Windows için kısıtlamalar nedeniyle otomatik olarak `False` olarak ayarlanır).
    *   **`drop_last=True`:** Son batch'in tam olmaması durumunda atılmasını sağlar, bu da model boyutları veya batch norm gibi bazı mimariler için daha kararlı eğitim sağlayabilir.
*   **GPU Bellek Yönetimi:**
    *   **`torch.cuda.empty_cache()`:** GPU üzerindeki önbelleği boşaltarak bellek fragmentasyonunu ve gereksiz bellek kullanımını azaltır. Her epoch sonunda çağrılır.
    *   **`gc.collect()`:** Python'ın çöp toplayıcısını manuel olarak tetikleyerek GPU belleğinin daha verimli serbest bırakılmasına yardımcı olur. Her epoch sonunda çağrılır.
    *   **`set_to_none=True` ile `optim.zero_grad()`:** Gradyan tensörlerini doğrudan sıfırlamak yerine `None` olarak ayarlayarak bellek tahsisini optimize eder.
*   **`ConvBlock` İyileştirmeleri (Dolaylı Etki):**
    *   **`GroupNorm`:** Daha küçük batch boyutlarında stabilite sağlar ve bazı GPU mimarilerinde performansı artırabilir.
    *   **`kernel_size=3`:** Daha az hesaplama gerektirir ve daha derin ağların daha verimli çalışmasına olanak tanır.
    *   **`SiLU()` aktivasyon fonksiyonu:** Diğer bazı aktivasyonlara göre daha düşük hesaplama maliyetine sahip olabilir.
    *   **`nn.Dropout2d` (Spatial Dropout):** Uzamsal bağıntıları koruyarak daha etkili bir düzenlileştirme ve genelleme sağlar, bu da daha az epokta iyi sonuçlar elde etmeye yardımcı olabilir.

## Veri Yükleme ve Ön İşleme (dataset/custom_dataset.py)

Veri seti işlemleri için özel bir PyTorch `Dataset` sınıfı (`CustomTumorDataset`) kullanılmıştır:

*   **`CustomTumorDataset`:** `.npy` formatındaki ön işlenmiş görüntülerden veri yükler. Bu, veri okuma hızını artırır.
*   **Görüntü Artırma ve Dönüşümler:** `Albumentations` kütüphanesi ile çeşitli rastgele dönüşümler uygulanır. Bu dönüşümler eğitim epoklarına göre dinamik olarak ayarlanır (`_get_current_augmenter` metodu):
    *   **Erken Eğitim Fazı (0-10 Epok):** Modelin daha çeşitli verilere maruz kalması için daha yüksek şiddetli artırmalar uygulanır. Bu aşama, modelin ilk öğrenme eğrisini hızlandırmasına yardımcı olur.
        *   `A.Rotate(limit=5, p=0.3)`
        *   `A.RandomRotate90(p=0.1)`
        *   `A.OneOf([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)], p=0.3)`
        *   `A.Affine(translate_percent={'x': 0.02, 'y': 0.02}, scale={'x': (0.96, 1.04), 'y': (0.96, 1.04)}, p=0.3)`
        *   `A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.3)`
        *   `A.GaussNoise(std_range=(0.005, 0.015), mean_range=(0.0, 0.0), p=0.1)`
    *   **Orta Eğitim Fazı (10-40 Epok):** Artırmaların şiddeti ve olasılıkları azaltılarak modelin daha oturmuş öğrenmesine olanak tanınır. Bu, modelin daha ince detayları öğrenmesine yardımcı olur.
        *   `A.Rotate(limit=3, p=0.1)`
        *   `A.RandomRotate90(p=0.05)`
        *   `A.OneOf([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)], p=0.1)`
        *   `A.Affine(translate_percent={'x': 0.005, 'y': 0.005}, scale={'x': (0.99, 1.01), 'y': (0.99, 1.01)}, p=0.1)`
        *   `A.RandomBrightnessContrast(brightness_limit=0.02, contrast_limit=0.02, p=0.1)`
        *   `A.GaussNoise(std_range=(0.003, 0.008), mean_range=(0.0, 0.0), p=0.05)`
    *   **Son Eğitim Fazı (40+ Epok):** Minimal artırmalar uygulanır, genellikle sadece temel dönüşümlerle modelin ince ayar yapmasına izin verilir. Bu aşama, modelin genelleme yeteneğini optimize etmeye odaklanır.
        *   `A.HorizontalFlip(p=1.0)`
    *   **Sabit Dönüşümler:** Tüm artırma aşamalarından sonra uygulanan sabit dönüşümler:
        *   `A.Resize(224, 224)`
        *   `A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))`
        *   `ToTensorV2()`

## Yardımcı Betikler

*   **`test.py`:** Eğitilmiş modelin test veri seti üzerindeki performansını değerlendirmek için kullanılır. Hem en iyi modeli hem de SWA modelini test edebilir.
*   **`losses.py`:** Gelecekte özel kayıp fonksiyonları eklemek için yer tutucu.
*   **Renk Kodlu Konsol Çıktısı:** Terminal çıktılarını renklendirmek için özel ANSI kaçış kodları kullanılır.
*   **Windows için Çoklu İşlem Desteği:** `multiprocessing` modülü, Windows işletim sistemlerinde uyumluluğu sağlamak için yapılandırılmıştır.

## Başlangıç

Projeyi çalıştırmak için gerekli bağımlılıkları yüklemeniz ve ardından `models/train.py` betiğini çalıştırmanız yeterlidir.


# Bağımlılıkları yükleyin 
pip install -r requirements.txt

# Modeli eğitin
python models/train.py

# Modeli test edin (eğitimden sonra)
python test.py
```

---

