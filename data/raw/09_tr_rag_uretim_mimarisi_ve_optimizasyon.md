# Üretim Ortamında Çok Dilli (TR+EN) RAG Mimarisi ve FinOps Optimizasyonu

Türkiye'deki ve çok dilli kurumsal ortamlardaki RAG (Retrieval-Augmented Generation) sistemleri, yalnızca İngilizce odaklı prototiplerde görünmeyen dilbilimsel ve maliyet kaynaklı mühendislik zorlukları barındırır. Bu kılavuz, Türkçe ve İngilizce dokümanlar üzerinde yüksek doğruluklu ve bütçe dostu bir Agentic RAG mimarisi tasarlarken uygulanması gereken temel pratikleri özetler.

## Türkçe Eklemeli Dil Yapısının Arama Motorlarına Etkisi

Türkçe, sondan eklemeli (agglutinative) bir dildir; tek bir kök sözcük onlarca farklı çekim ve yapım ekiyle görünebilir (örneğin `vektör`, `vektörlerin`, `vektörleştirme`, `vektörlerindeki`). Bu durum retrieval katmanında iki kritik sonuç doğurur:

1. **Saf BM25 Sınırlılığı**: Türkçe için özel morfolojik kök bulucu (lemmatizer/stemmer) yoksa standart boşluk/noktalama tabanlı BM25 tokenizasyonu `vektör` ile `vektörlerin` terimlerini farklı token'lar olarak görür. Buna karşılık teknik parametre adları (`ef_construct`, `work_mem`, `gpu_memory_utilization`) ve İngilizce terimler BM25 tarafından birebir yüksek hassasiyetle yakalanır.
2. **Çok Dilli Dense + Cross-Encoder Gücü**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (768 boyut) veya `intfloat/multilingual-e5-large` (1024 boyut) gibi çok dilli gömme (embedding) modelleri, eklemeli Türkçe sorgular ile İngilizce teknik dokümanları **aynı anlamsal vektör uzayına** haritalar. İkinci aşamada `BAAI/bge-reranker-v2-m3-int8` Cross-Encoder kullanımı, çapraz dilli (TR sorgu → EN doküman) sıralama doğruluğunu belirgin biçimde yükseltir.

## Katmanlı LLM Yönlendirmesi (Tiered FinOps Routing)

Her RAG adımında en pahalı amiral gemisi modeli kullanmak maliyeti gereksiz yere 5–10 kat artırır. Üretim sistemlerinde görev zorluğuna göre **iki katmanlı model mimarisi** önerilir:

| Görev Katmanı | Sorumlu Adımlar | Önerilen Model Profili | Maliyet / Gecikme Etkisi |
| :--- | :--- | :--- | :--- |
| **Hızlı / Ekonomik Katman (`LLM_FAST_MODEL`)** | Sorgu Sınıflandırma (Router), Sorgu Yeniden Yazma, Multi-Query Ayrıştırma, CRAG Grader | `qwen/qwen3-30b-a3b-instruct-2507` (MoE, yalnızca 3B aktif parametre) | Çok düşük token maliyeti (~\$0.10/1M giriş), hızlı yapısal JSON yanıtı |
| **Güçlü Sentez Katmanı (`LLM_GENERATION_MODEL`)** | Kaynaklı Yanıt Üretimi (Grounded Synthesis), Katman-2 Atıf Doğrulama (Entailment) | `deepseek/deepseek-v3.2` | Yüksek akıl yürütme ve çok dilli atıf sadakati (~\$0.26/1M giriş, ~\$0.38/1M çıkış) |
| **Yerel Sıfır Maliyet Katmanı (ONNX)** | Dense Embedding, BM25 Sparse, Cross-Encoder Reranking | FastEmbed ONNX (`$0` API maliyeti) | API kotası tüketmez, veri dışarı çıkmaz |

### Maliyet Koruma Mekanizmaları
- **Yerel Prompt Önbelleği (`SQLiteLLMCache`)**: Aynı `(model, temperature, messages)` kombinasyonu geliştirme ve benchmark tekrarlarında yerel SQLite önbelleğinden `0 ms` ağ maliyeti ve `$0` ücretle döner.
- **Sert Bütçe Koruması (`CostTracker` & `MAX_RUN_COST_USD`)**: Her LLM yanıtındaki `prompt_tokens` ve `completion_tokens` verisi gerçek zamanlı olarak USD maliyetine çevrilir; kümülatif harcama belirlenen eşiği (örneğin `0.50 USD`) aştığında `BudgetExceededError` fırlatılarak kontrolden çıkan döngüler durdurulur.
- **Konuşma (Conversational) Kestirmesi**: "Merhaba", "Teşekkürler" veya önceki yanıtı özetleme gibi doküman gerektirmeyen sorgularda Qdrant ve Reranker tamamen atlanarak hem gecikme hem bağlam token maliyeti sıfırlanır.

## Halüsinasyon Azaltma ve Dil Tutarlılığı Kontrol Listesi

Üretim RAG uygulamalarında kullanıcı Türkçe soru sorduğunda kaynak doküman İngilizce olsa dahi yanıtın **kullanıcının dilinde (Türkçe)** üretilmesi, ancak teknik terim ve parametre adlarının orijinal yazımının (`ef_construct`, `Fusion.RRF`, `PagedAttention`) korunması gerekir. Ek olarak:
- Her teknik iddia mutlaka `[C1]` veya `[W1]` şeklinde satır içi kaynak etiketi taşımalıdır.
- Sistem prompt'u modele "yalnızca sağlanan bağlamdaki bilgiyi kullan, bağlamda yoksa açıkça belirt" kuralını katı biçimde dikte etmelidir.
- Katman-1 deterministik atıf kontrolü, bağlamda bulunmayan hayali kaynak ID'lerini (`[C9]` gibi) sıfır maliyetle anında yakalar.
