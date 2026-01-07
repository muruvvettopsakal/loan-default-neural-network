Loan Default Prediction Using Neural Networks
1️⃣ Veri Seti Açıklaması

Bu projede, Kaggle Loan Default Dataset kullanılmıştır. Veri seti, bireylerin demografik bilgileri, finansal durumları ve kredi özelliklerine göre kredi temerrüt (default) riskinin tahmin edilmesini amaçlamaktadır.

Toplam gözlem sayısı: 255.347

Toplam özellik sayısı: 18

Hedef değişken: Default

0 → Kredi geri ödemesi yapılmış

1 → Kredi temerrüde düşmüş

Veri Setinde Bulunan Bazı Özellikler:

Age (Yaş)

Income (Gelir)

LoanAmount (Kredi Tutarı)

CreditScore (Kredi Skoru)

MonthsEmployed (Çalışma Süresi)

InterestRate (Faiz Oranı)

DTIRatio (Borç / Gelir Oranı)

Education (Eğitim Durumu)

EmploymentType (İstihdam Türü)

MaritalStatus (Medeni Hal)

LoanPurpose (Kredi Amacı)

Kategorik değişkenler One-Hot Encoding yöntemiyle sayısal hale getirilmiştir.
Sayısal değişkenler StandardScaler kullanılarak ölçeklendirilmiştir.

2️⃣ Model Mimarisi

Bu çalışmada, Scikit-learn kütüphanesinde bulunan MLPClassifier kullanılarak çok katmanlı yapay sinir ağı (Artificial Neural Network) modeli oluşturulmuştur.

Model Yapısı:

Giriş Katmanı: Veri setindeki özellik sayısına göre otomatik

Gizli Katmanlar:

gizli katman → 64 nöron

gizli katman → 32 nöron

Aktivasyon Fonksiyonu: ReLU

Çıkış Katmanı:

1 nöron

Binary sınıflandırma (Default / Non-default)

Eğitim Parametreleri:

Optimizer: Adam

Maksimum iterasyon (epoch): 20

Eğitim/Test ayrımı: %80 / %20

Özellik ölçekleme: StandardScaler

Model, kredi temerrüt tahminini ikili sınıflandırma problemi olarak ele almaktadır.

3️⃣ Eğitim Grafikleri ve Başarı Metrikleri
🔹 Eğitim Kayıp (Loss) Grafiği

Aşağıdaki grafikte, modelin eğitim süreci boyunca kayıp (loss) değerinin iterasyonlara göre değişimi gösterilmektedir:

📊 Training Loss Curve

<img width="477" height="360" alt="Ekran görüntüsü 2026-01-08 012119" src="https://github.com/user-attachments/assets/0dcdc21f-e2b4-4bd5-b9e9-9321b573283d" />

Bu grafik, modelin öğrenme sürecini ve optimizasyon davranışını görsel olarak sunmaktadır.

🔹 Confusion Matrix

Modelin test verisi üzerindeki sınıflandırma performansı aşağıdaki confusion matrix ile gösterilmiştir:

📊 Confusion Matrix


🔹 Başarı Metrikleri

Accuracy:
<img width="479" height="359" alt="Ekran görüntüsü 2026-01-08 012040" src="https://github.com/user-attachments/assets/d08465d0-bb0e-4053-a955-f01fa2736cb8" />

0.8858


Classification Report (Özet):

Default olmayan müşteriler yüksek doğrulukla tahmin edilmiştir.

Default sınıfında, veri dengesizliği nedeniyle recall değeri görece düşüktür.

Model, kredi risk analizi açısından güvenilir sonuçlar üretmektedir.

4️⃣ Kullanılan Teknolojiler

Python

Pandas

NumPy

Scikit-learn

Matplotlib

5️⃣ GitHub Repository

🔗 GitHub Linki:
👉(https://github.com/muruvvettopsakal/loan-default-neural-network)

6️⃣ Sonuç

Bu projede, yapay sinir ağları kullanılarak kredi temerrüt tahmini başarıyla gerçekleştirilmiştir. Model, gerçek dünya bankacılık ve finans uygulamalarında kullanılan kredi risk analizine uygun sonuçlar sunmaktadır.
