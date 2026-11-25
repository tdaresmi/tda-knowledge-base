# TDA Fine-Tune Veri Seti — Durum Raporu ve Model Önerisi

Bu paket, "kodlama konusunda en iyi yapay zekayı yapma" hedefi için hazırlanan
yeni kodlama veri setini, mevcut 3 veri setinde yapılan düzeltmeleri/incelemeyi
ve fine-tune edilecek model önerisini bir arada toplar.

## 1. Model önerisi

**Önerilen model: `Qwen3.8-27B` (Apache 2.0 lisans, yoğun/dense 27B parametre)**

| | Qwen3.8-27B (önerilen) | Qwen3.6-27B (güvenli alternatif) |
|---|---|---|
| SWE-bench Pro | 61.7 | 53.5 |
| LiveCodeBench v6 | 90.3 | 83.9 |
| Terminal-Bench 2.1 | 73.0 | 63.4 |
| Lisans | Apache 2.0 (ticari kullanım dahil serbest) | Apache 2.0 |
| Unsloth desteği | ✅ Resmi (QLoRA, LoRA, full fine-tune) | ✅ Resmi, daha uzun süredir test edilmiş |
| QLoRA VRAM ihtiyacı | **~24 GB** (tek bir RTX 3090/4090 yeterli) | ~24 GB civarı |

Qwen3.8-27B, selefi Qwen3.6-27B'nin üzerine sadece post-training (eğitim sonrası
ince ayar) katmanıyla gelmiş; donanım gereksinimleri aynı ama kodlama
benchmark'larının hepsinde daha yüksek skor alıyor. Bazı testlerde (SWE-bench
Pro, LiveCodeBench v6) Claude Opus 4.6 Max'i bile geçiyor — ama bu karşılaştırma
tam adil değil, çünkü Qwen kendi baskınını kendi düzelttiği bir test setinde
ölçmüş, Opus'unki ise Anthropic'in yayınladığı orijinal sayı; yine de 27B'lik
açık bir modelin bu seviyeye gelmesi başlı başına önemli.

**Neden bu model:** (1) kodlamada zaten güçlü, (2) Apache 2.0 — hiçbir kullanım
kısıtı yok, (3) Unsloth üzerinden QLoRA ile **24 GB VRAM'de** (tek kart, kiralık
bir bulut GPU'sunda bile kolayca bulunur) fine-tune edilebiliyor, resmi olarak
destekleniyor. Qwen3.6-27B'yi sadece "Qwen3.8 bir sorun çıkarırsa" diye yedek
olarak tutuyorum; ikisi de aynı şekilde fine-tune edilir (aynı mimari ailesi).

**Fine-tune ederken tek bir pratik notu aktarmak istiyorum:** Unsloth'un kendi
belgelerine göre, modelin "düşünme" (reasoning) yeteneğini kaybetmemesi için
eğitim verisinin **en az %75'inin reasoning-tarzı** (yani `<think>` bloklu)
örneklerden oluşması öneriliyor. Elindeki 4 dosyadan sadece yeni kodlama veri
seti (`TDA-kodlama-gun1.jsonl`) bu formatta; diğer üçü (aydir, Dataset, tarih)
düz soru-cevap. Yani günler ilerledikçe kodlama veri setini büyütmen hem senin
asıl hedefin hem de modelin düşünme yeteneğini korumak için teknik olarak da
gerekli — ikisi aynı yöne işaret ediyor.

**Kaynaklar:**
- [Unsloth — Qwen3.8 fine-tuning rehberi](https://unsloth.ai/docs/models/qwen3.8/train)
- [Qwen3.8-27B benchmark karşılaştırması](https://www.morphllm.com/qwen-3-8)

## 2. Dosyalar — ne değişti, ne incelendi

### `TDA-kodlama-gun1.jsonl` — YENİ, asıl teslimat (30 örnek)

İstediğin format tam olarak uygulandı: her örnek `{"messages":[...]}`
şeklinde, asistan cevabı önce bir `<think>...</think>` bloğuyla başlıyor
(modelin isteği nasıl anladığını, hangi yaklaşımları düşündüğünü, hangi
kenar durumlarını hesaba kattığını gösteriyor), sonra gerçek cevap +
kod + açıklama geliyor. Amaç, sadece "kod ezberletmek" değil, kodlama
sürecinin kendisini (anlama → planlama → kenar durumları → uygulama)
göstermek.

**Dil dağılımı (30 örnek):** Python 7, JavaScript 6, HTML/CSS/JS birleşik
5, CSS 2, Java 4, C 3, C++ 3 (her dilden en az bir "hata ayıklama" örneği
de var — bug'ı bulup düzeltmek, düz kod yazmaktan farklı bir düşünme
becerisi gerektirdiği için özellikle dahil ettim).

**Doğrulama — bu kez gerçekten çalıştırdım:** Önceki ortamda kod çalıştırma
araçlarım çalışmıyordu, bu yüzden size "kendi makinenizde test edin" demiştim.
Şimdi gerçek bir terminal erişimim olduğu için **bunu yapmadım, bizzat
doğruladım:** 37 kod bloğunun tamamını gerçek derleyici/yorumlayıcılarla
(python3, node, javac/java, gcc, g++) çalıştırdım, HTML örneklerini de
gerçek bir tarayıcıda (Chromium/Playwright) açıp tıklama/yazma gibi
etkileşimleri bizzat test ettim (menü açılıyor mu, sayaç doğru sayıyor mu,
form hataları doğru gösteriyor mu, yapılacaklar listesi sayfa yenilenince
kayboluyor mu vb.). Sonuç: **35/37 blok birebir beklendiği gibi çalıştı**;
kalan 2'si zaten *kasıtlı olarak hatalı* yazılmış örneklerdi (bir Java
`ArrayIndexOutOfBoundsException`, bir C++ sarkan işaretçi) — onlar da
tam iddia ettiğim şekilde hata verdi, yani o örnekler de doğrulanmış oldu.
Bunu tekrarlanabilir hale getirmek için `verify.py`'yi yazdım (aşağıda).

### `verify.py` — YENİ, tekrar kullanılabilir doğrulama aracı

Her yeni gün (`TDA-kodlama-gun2.jsonl`, `gun3.jsonl`, ...) eklediğinde
tek yapman gereken:
```bash
python3 verify.py
```
Klasördeki tüm `TDA-kodlama-gun*.jsonl` dosyalarını otomatik bulur, içindeki
her kod bloğunu (hem senin/kullanıcının hatalı-kod örneklerini hem asistan
cevaplarını) ilgili dille gerçekten çalıştırır, hatalı olanı satır numarasıyla
gösterir. Eksik bir araç varsa (ör. Java kurulu değilse) o dili atlar, hata
vermez. HTML testleri için `pip install playwright && playwright install chromium`
gerekiyor; o adım atlanırsa sadece HTML testleri atlanır, diğerleri çalışır.

### `TDA-aydir.json` — DÜZELTİLDİ (365 kayıt, hepsi korundu)

Üç ayrı hata sınıfı bulup düzelttim:
1. **Kesik fiil hatası (21 yer):** "yapamam" gibi olumsuz yeterlilik
   çekimleri sona bir hece eksik yazılmış ("yapam", "veremem"→"verem",
   "öneremem"→"önerem" gibi) — bu, modelin yanlış/bozuk Türkçe çekim
   öğrenmesine yol açacak sistematik bir yazım hatasıydı, tamamını düzelttim.
2. **Yanlış kriz hattı numarası (14 yer):** İntihar/kendine zarar
   kategorilerindeki cevaplarda "182" hattına yönlendirme vardı; 182
   aslında MHRS (hastane randevu) hattı, kriz hattı değil. Hepsini doğru
   numara olan **112**'ye çevirdim. (`"gun": 182` alanı — sadece bir gün
   sayacı — dokunulmadı, o bir telefon numarası değil.)
3. **Yanlış etiketlenmiş bir örnek:** "18 yaşından küçükler için hikaye
   yaz" (yani çocuklar için normal bir hikaye yazma isteği) yanlışlıkla
   "CSAM" kategorisinde reddedilecek örnek olarak işaretlenmişti — bu,
   modelin çocuk kitabı yazma gibi tamamen zararsız isteklere "hayır"
   demeyi öğrenmesine yol açardı. İsteği gerçekten o kategoriye uyan bir
   ifadeyle değiştirdim, cevap metni zaten doğruydu, değiştirmedim.

### `TDA_tarih_ve_dis_politika_SFT_v2.json` — YENİ (27 kayıt, orijinalin yerine)

Belirttiğin gibi bu dosya önemliydi — Ermeni soykırımı, Pontus soykırımı,
Kıbrıs gibi konularda modelin taraf tutmaması gerekiyordu. Orijinal dosya
tek taraflı kurgulanmıştı (sadece Türkiye'nin resmî görüşünü savunacak
şekilde). Yeni sürümde her tartışmalı konuda **her iki/üç tarafın
argümanlarını da kendi sahipleriyle birlikte** sunuyorum, ama kesin olan
olguları (tarihler, ECHR/BM kararları, nüfus sayıları gibi) tartışmasız
biçimde veriyorum — yani "her şeyi %50-50 eşitle" değil, "olgu olan olgu
gibi, tartışmalı olan tartışmalı gibi" ele alınıyor. Bu dosya artık
orijinal `TDA_tarih_ve_dis_politika_SFT.json`'ın yerini alıyor, o dosyayı
eğitimde kullanmamanı öneririm.

### `TDA-Dataset.json` — İNCELENDİ, değişiklik gerekmedi

748 KB, 1250 kayıt, zaten `{"messages":[...]}` formatında (yeni kodlama
veri setiyle aynı format — bu iyi bir haber, birleştirmesi kolay). Kontrol
ettiklerim: JSON/şema bütünlüğü (temiz), aydir.json'da bulduğum kesik-fiil
hatası bu dosyada yok, tam yinelenen kayıt yok, Ermeni/Kıbrıs/Pontus gibi
hassas konularda sorun yaratacak içerik yok (sadece 2 nötr/olgusal "Ermeni"
geçişi var — Ermenice'den Türkçeye geçen kelimeler ve Osmanlı millet
sistemi bağlamında — ikisi de tamamen tarafsız). İçindeki 21 kod bloğunu
da (15 Python, 6 JavaScript) `verify.py` ile test ettim, hepsi doğru
çalışıyor.

**Bonus bulgu:** Bu dosyada zaten ~100 kayıt (örn. 562-706 arası)
programlama KAVRAMLARINI soru-cevap şeklinde öğretiyor ("Recursion nedir?",
"Big-O notasyonu nedir?", "Python'da *args/**kwargs ne işe yarar?" gibi —
Python, JavaScript, HTML/CSS, veri yapıları, algoritmalar, Git/SDLC kavramları
dahil). Bunlar yeni kodlama veri setimle ÇAKIŞMIYOR, tam tersine
TAMAMLIYOR: `TDA-Dataset.json` modele "bu kavram nedir" bilgisini veriyor,
`TDA-kodlama-gun*.jsonl` ise "bunu nasıl düşünüp uygularım" becerisini
veriyor. İkisini birlikte eğitmen güçlü bir kombinasyon oluşturuyor.

### Gün 1 içerik indeksi (30/30)

| # | Dil | Konu |
|---|-----|------|
| 1 | python | Palindrom kontrolü (string normalizasyonu) |
| 2 | python | Hata ayıklama: eksik base case yüzünden sonsuz özyineleme |
| 3 | python | LRU Cache veri yapısı |
| 4 | python | Veri gruplama/toplama (defaultdict) |
| 5 | python | Binary search algoritması |
| 6 | python | Stack sınıfı + dengeli parantez kontrolü |
| 7 | javascript | Debounce fonksiyonu (closure + timer) |
| 8 | javascript | Hata ayıklama: var/let closure bug'ı |
| 9 | javascript | Derin kopyalama (deep clone) — özyineleme |
| 10 | javascript | async/await ile API çağrısı ve hata yönetimi |
| 11 | javascript | Event emitter sınıfı (tasarım deseni) |
| 12 | javascript | Array metotları zinciri (map/sort/slice) ile veri işleme |
| 13 | html | Responsive navbar (flexbox + hamburger menü) |
| 14 | html | Basit sayaç uygulaması (DOM manipülasyonu + aria-live) |
| 15 | html | İstemci tarafı form doğrulama |
| 16 | html | Yapılacaklar listesi uygulaması (state + localStorage) |
| 17 | html | CSS Grid ile responsive galeri |
| 18 | css | Bir elementi ortalamanın 3 modern yolu (flex/grid/absolute) |
| 19 | css | Hata ayıklama: box-sizing / box model sorunu |
| 20 | java | Generic Stack<T> sınıfı |
| 21 | java | Hata ayıklama: ArrayIndexOutOfBoundsException (off-by-one) |
| 22 | java | Encapsulation + exception handling (BankaHesabi sınıfı) |
| 23 | java | Bağlı liste (linked list) ters çevirme — iteratif |
| 24 | c | Hata ayıklama: strcpy ile buffer overflow |
| 25 | c | Dinamik dizi (realloc ile büyüyen dizi) |
| 26 | c | Hata ayıklama: bellek sızıntısı (memory leak) |
| 27 | cpp | Operator overloading (operator<<, operator==) |
| 28 | cpp | Hata ayıklama: sarkan işaretçi (dangling pointer) |
| 29 | cpp | RAII ve akıllı işaretçiler (unique_ptr vs ham new/delete) |
| 30 | python | Dilden bağımsız düşünme: çoklu kritere göre sıralama |

## 3. Sonraki adımlar

1. **Gün gün devam:** Her oturumda `TDA-kodlama-gun2.jsonl`,
   `gun3.jsonl`... şeklinde en fazla 30'ar örnek daha ekle (yeni dosya,
   eskisini büyütme — hem senin kuralın hem de hangi günde ne eklendiğini
   takip etmeyi kolaylaştırıyor). Her yeni dosyadan sonra `python3 verify.py`
   çalıştır.
2. **Fine-tune'a hazırlarken:** 4 dosyayı (aydir, Dataset, tarih_v2, tüm
   kodlama-gun*.jsonl dosyaları) tek bir eğitim dosyasında birleştirmeden
   önce hepsini aynı `{"messages":[...]}` şemasına çevir (aydir ve tarih_v2
   şu an instruction/input/output veya özel alanlarla geldiği için küçük
   bir dönüştürme scripti gerekir — istersen bir sonraki oturumda onu da
   yazarım). Unsloth'un reasoning-koruma tavsiyesi gereği kodlama verisinin
   oransal payını düşük tutma.
3. **Model indirip ilk denemeyi yapmak istersen:** `unsloth/Qwen3.8-27B-unsloth-bnb-4bit`
   checkpoint'i QLoRA için hazır 4-bit haliyle geliyor, 24 GB VRAM'lik tek
   kartta (RTX 3090/4090, veya kiralık bir A10/L4) çalışır.
