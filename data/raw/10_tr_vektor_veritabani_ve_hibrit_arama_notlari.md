# Vektör Veritabanı Pratikleri, Yapısal Chunking ve Dinamik Top-K Tasarımı

Bir RAG sisteminin kalitesi çoğu zaman kullanılan LLM'den önce **parçalama (chunking)** ve **aday seçim (retrieval + dynamic top-k)** stratejileri tarafından belirlenir. Kötü parçalanmış bir dokümanı ya da alakasız gürültüyle şişirilmiş bir bağlamı en güçlü LLM bile güvenilir bir yanıta dönüştüremez.

## Yapısal Farkındalıklı Parçalama (Structure-Aware Chunking)

Sabit karakter uzunluğuna göre körlemesine yapılan parçalama (örneğin her 500 karakterde kesme), Markdown tablolarını ortasından böler, kod bloklarını anlamsız parçalara ayırır ve başlık bağlamını yok eder. Yapısal parçalayıcı şu dört kuralı uygular:

1. **Başlık İz Yolu (`section_path`)**: Her parça, ait olduğu doküman ve alt başlık hiyerarşisini (`Doküman > H1 > H2 > H3`) metadata olarak taşır ve gömme (embedding) aşamasında metne bağlam başlığı olarak eklenir.
2. **Kod Bloğu ve Tablo Bütünlüğü**: ` ``` ` ile çevrili kod blokları ve Markdown tabloları atomik blok kabul edilir; tek başına maksimum token sınırını aşmadıkça bölünmezler. Ayrıca kod içindeki `# yorum` satırları kesinlikle Markdown başlığı sanılmaz.
3. **Token Bütçeli Örtüşme (Token-Aware Overlap)**: Karakter yerine `tiktoken` (`cl100k_base`) ile token sayılır; varsayılan `max_tokens=512` ve `overlap_tokens=64` yalnızca aynı bölüm içindeki ardışık parçalar arasında uygulanır, ilgisiz ana bölümler birbirine sızdırılmaz.
4. **Deterministik `chunk_id` (Idempotent İndeksleme)**: Her parça için `UUIDv5(doc_id, chunk_index, content_hash)` üretilir. Böylece `make ingest` komutu tekrar çalıştırıldığında Qdrant'ta mükerrer (duplicate) kayıt oluşmaz, aynı noktalar temiz biçimde güncellenir.

## Üç Aşamalı Dinamik Top-K Kesme Algoritması

Sabit `top_k=5` yaklaşımı iki uçta da suboptimaldir:
- Tek bir parametreyi soran **olgusal (factual)** sorguda 2. ve 3. sıradan sonraki parçalar gürültüdür ("lost in the middle" riski + gereksiz token maliyeti).
- Birden fazla mimariyi kıyaslayan **analitik (analytical)** sorguda ise 5 parça tüm boyutları kapsamaya yetmeyebilir (`Context Recall` düşer).

Cross-Encoder reranker'ın sigmoid ile `[0, 1]` aralığına kalibre edilmiş skorları $s_1 \ge s_2 \ge \dots \ge s_n$ üzerinde çalışan **3 aşamalı Dinamik Top-K** bu sorunu çözer:

1. **Aşama 1 — Mutlak Relevans Tabanı (`tau_min`)**: `s_i < tau_min` olan tüm düşük kaliteli adaylar elenir (olgusal için `0.35`, analitik için `0.30`).
2. **Aşama 2 — Göreli Düşüş (`alpha_relative`) ve Dirsek (Elbow `delta_gap`) Tespiti**:
   - Skor, en iyi sonucun belli bir oranının altına düştüğünde ($s_i < \alpha \cdot s_1$) veya ardışık iki parça arasında keskin bir kalite uçurumu oluştuğunda ($s_{i-1} - s_i > \Delta_{\text{gap}}$) kesme yapılır.
   - Olgusal sorgularda agresif kesim (`alpha=0.65`, `delta_gap=0.25`) uygulanırken analitik sorgularda daha kapsayıcı kesim (`alpha=0.45`, `delta_gap=0.35`) seçilir.
3. **Aşama 3 — Alt/Üst Sınır ve Token Bütçesi Kıskacı**: Sonuç sayısı `[rerank_min_k, rerank_max_k]` aralığında ve toplam bağlam `max_context_tokens` bütçesi içinde tutulur (ancak `tau_min` altındaki tamamen alakasız parçalar sırf `min_k` dolsun diye zorla eklenmez; böylece düşük güven tespiti bozulmaz).

## Qdrant ve pgvector Seçim Özeti

Küçük ölçekli (< 1M vektör) ve yoğun ilişkisel `JOIN` gerektiren uygulamalarda `pgvector` tek veritabanı sadeliği sunar. Buna karşılık hibrit arama (Dense + BM25 `Modifier.IDF` + sunucu tarafı `RRF`), seçici metadata filtrelerinde yüksek recall (`Filterable HNSW`), dahili `int8`/binary nicemleme (`quantization`) ve yatay ölçekleme gereken uzman Agentic RAG sistemlerinde **Qdrant** hem gecikme hem geliştirici ergonomisi açısından belirgin üstünlük sağlar.
