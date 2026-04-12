# Rencana Modifikasi Konservatif F5-TTS: Mengganti 1–2 Blok Transformer dengan Mamba SSM v2.3.1 (Hybrid DiT) dari Checkpoint Indo Eempostor

## Ringkasan eksekutif

Tujuan Anda—mengganti **hanya 1–2 blok** pada backbone **Diffusion Transformer (DiT)** F5-TTS dengan **Mamba SSM (Mamba2) v2.3.1**—paling aman bila dilakukan sebagai *surgical swap* yang **mempertahankan antarmuka blok, dimensi tensor, pola residual + gating**, serta meminimalkan jumlah langkah training tambahan. Dalam kode resmi F5-TTS, backbone DiT terdiri dari *ModuleList* `transformer_blocks` berisi `DiTBlock` homogen yang menerima `(x, t, mask, rope)`; setiap `DiTBlock` melakukan **AdaLayerNorm-modulated self-attention** lalu **MLP** dengan residual dan **gating** (mirip *adaLN*-style) sehingga titik penggantian paling konservatif adalah **mengganti “self-attention submodule”** di 1–2 `DiTBlock` terpilih tanpa menyentuh `AdaLayerNorm`, `ff_norm`, maupun `ff` yang sudah terlatih. citeturn10view1turn13view0turn37view0

Rekomendasi inti:

1. **Mulai dari 1 blok** dulu (mis. indeks tengah dari `depth=22` → kira-kira `block_id = 11`) agar risiko degradasi kecil dan debugging mudah; baru setelah stabil, lanjut ke 2 blok (mis. `block_id ∈ {7, 14}` sebagai sebaran). Semua kandidat tetap kompatibel karena `DiTBlock` homogen dan ber-dimensi sama. citeturn18view0turn10view1turn13view0turn37view0  
2. Implementasikan penggantian sebagai **mixer drop-in**: `attn(x_norm, mask, rope)` → `mamba(x_norm, mask)` lalu tetap pakai `gate_msa` dari `AdaLayerNorm` dan jalur MLP yang sama. Ini meminimalkan perubahan perilaku di luar komponen “mixing across time”. citeturn13view0turn12view0  
3. Untuk kompatibilitas tugas **speech infilling** (butuh konteks kiri+kanan), gunakan **BiMamba2** (dua Mamba2: forward + reverse, lalu digabung) atau jika sangat konservatif terhadap compute, gunakan Mamba2 kausal tetapi batasi penggantian hanya 1 blok dan sandingkan dengan distilasi kuat. (F5-TTS sendiri melakukan attention non-causal pada `scaled_dot_product_attention(... is_causal=False)`.) citeturn13view0turn30view3  
4. Distilasi paling hemat waktu adalah **distilasi lokal per-blok** dengan *teacher* berupa attention yang sudah ada di checkpoint (dibekukan) sehingga Anda **tidak perlu forward teacher penuh**. Anda melatih Mamba untuk mereplikasi **output attention** pada blok yang sama, kemudian *fade-in* Mamba (α naik) sampai attention bisa dimatikan.  
5. Training protokol konservatif: gunakan **AdamW + warmup linear + decay linear** (konsisten dengan paper dan trainer resmi), batch berbasis *frames*, checkpoint per beberapa ribu update, dan early stopping berbasis metrik ringan (UTMOS + WER + SIM + loss distilasi). citeturn37view0turn23view2turn33view0  

Variabel yang sengaja dibiarkan *configurable* (karena Anda menyatakan detail tidak diketahui):  
`BLOCK_IDS` (indeks blok yang diganti), `N_UPDATES_STAGE*`, `BATCH_FRAMES_PER_GPU`, `GPU_TYPE`, `DATA_SUBSET_HOURS`, `SEQ_LEN_FRAMES`, dan apakah checkpoint Indo berisi `ema_model_state_dict` atau hanya `model_state_dict`. (Kode load checkpoint resmi mendukung keduanya via flag `use_ema`.) citeturn26view2turn38view1  

## Konteks dan ringkasan arsitektur F5-TTS yang relevan untuk penggantian blok

F5-TTS adalah sistem TTS non-autoregresif berbasis **Conditional Flow Matching (CFM)** dengan backbone **DiT**; training memprediksi **flow** `x1-x0` dari campuran `φ=(1-t)x0 + t x1` dan mengoptimasi **MSE** hanya pada span yang di-*mask* (text-guided speech infilling). Ini terlihat langsung di implementasi `CFM.forward`: membentuk `x0` Gaussian, sampling `time`, menghitung `φ` dan `flow`, lalu `pred = transformer(...)` dan `loss = mse(pred, flow)` yang diambil pada `rand_span_mask`. citeturn29view0turn21view0turn21view1  

### Struktur backbone DiT di kode resmi

Backbone `DiT` (F5-TTS) melakukan:

- `time_embed(time)` menghasilkan embedding `t` (Sinusoidal → MLP). citeturn13view0turn37view0  
- `InputEmbedding`: menggabungkan **noised audio** `x`, **cond audio** `cond` (masked), dan `text_embed`, lalu linear projection dan menambahkan **convolutional position embedding** (`ConvPositionEmbedding`) dengan dukungan `mask`. citeturn9view0turn30view2  
- Menyusun `transformer_blocks = ModuleList([DiTBlock(...) for _ in range(depth)])` dan menjalankan berurutan; `rope = rotary_embed.forward_from_seq_len(seq_len)` dibuat sekali dan dikirim ke tiap blok. citeturn10view1turn10view4turn13view0  
- `DiTBlock` mengandung `AdaLayerNorm` untuk memproduksi modulasi + gate, kemudian `Attention`, lalu MLP dan residual. Gating utamanya: `x = x + gate_msa * attn_output`, dan `x = x + gate_mlp * ff_output`. citeturn13view0turn12view0  

### Konfigurasi dimensi (Base) yang menentukan kompatibilitas Mamba

Untuk konfigurasi Base (yang selaras dengan paper base model dan config YAML):

- `dim = 1024`, `depth = 22`, `heads = 16`, `ff_mult = 2`, `text_dim = 512`, `conv_layers = 4`, `pe_attn_head = 1`, dropout `0.1`. citeturn18view0turn37view0  
- Paper menyatakan base model DiT: **22 layer, 16 head, 1024/2048 dim (embed/FFN)** plus ConvNeXt V2 4 layer (512/1024) total **335.8M parameter**; optimizer **AdamW** peak LR `7.5e-5`, warmup `20K`, linear decay; grad clip 1. citeturn37view0  

Implikasi untuk penggantian: setiap kandidat blok yang diganti harus menjaga tensor `x` dengan shape `[B, N, 1024]` agar jalur residual dan MLP di blok tersebut tetap valid. Dengan kata lain, target paling konservatif adalah mengganti submodule `self.attn` di `DiTBlock` dengan modul yang **input-output dimensinya identik**.

## Kandidat blok transformer yang diganti dan justifikasinya

Di F5-TTS, `DiTBlock` di `transformer_blocks` bersifat **homogen** (parameterisasi sama) sehingga kompatibilitas dimensi dan pola residual sama untuk semua indeks. citeturn10view1turn13view0  
Namun, pemilihan indeks tetap memengaruhi risiko karena kedekatan dengan embed input dan proyeksi output. Pendekatan konservatif biasanya menghindari “edge layers” pada eksperimen pertama.

### Kandidat paling konservatif

Berikut tabel kandidat yang memprioritaskan *minimal disruption* dan *minimal training time* (mulai dari 1 blok):

| Kandidat `BLOCK_IDS` | Mengapa kompatibel | Risiko kualitas | Alasan konservatif | Catatan implementasi |
|---|---|---|---|---|
| `{⌊depth/2⌋}` (mis. `{11}` untuk `depth=22`) | Semua `DiTBlock` identik: in/out `dim=1024`, residual+gating sama. citeturn13view0turn10view1 | Rendah–sedang | Efek perubahan terlokalisasi; layer lain tetap menyediakan global mixing via attention | Cocok untuk uji awal + distilasi lokal |
| `{depth-3}` (mis. `{19}`) | Sama seperti di atas | Sedang | Dekat output; jika mismatch dapat lebih terlihat pada *mel* | Pilih jika Anda ingin efek lebih “langsung” namun butuh KD lebih ketat |
| `{2}` atau `{3}` | Sama | Sedang–tinggi | Early layers memproses sinyal lebih “raw”; gangguan bisa menyebar | Hanya disarankan jika sudah stabil dengan kandidat tengah |

### Kandidat untuk 2 blok (setelah 1 blok berhasil)

| Kandidat `BLOCK_IDS` | Rasional | Risiko | Catatan |
|---|---|---|---|
| `{⌊depth/3⌋, ⌊2depth/3⌋}` (mis. `{7,14}`) | Sebaran perubahan, tidak menumpuk di satu wilayah representasi | Sedang | Cocok untuk menguji “efek kumulatif” tanpa mendekati layer 0 / last |
| `{⌊depth/2⌋-1, ⌊depth/2⌋+1}` (mis. `{10,12}`) | Menguji “local cluster” | Sedang–tinggi | Bila salah, degradasi cenderung lebih besar; butuh KD lebih lama |

**Kriteria kompatibilitas teknis yang wajib dipenuhi** (untuk semua kandidat):  
(1) `x` tetap `[B,N,dim]`, (2) tetap memakai `AdaLayerNorm` yang menghasilkan `gate_msa, shift/scale`, (3) tetap menjaga residual `x + gate * out`, (4) tidak mengubah kontrak `forward(x,t,mask,rope)`. Pola ini jelas di `DiTBlock.forward`. citeturn13view0turn12view0  

## Strategi integrasi Mamba SSM v2.3.1 yang menjaga arsitektur asli

### Mengapa Mamba2 (SSM) dan kenapa v2.3.1

Paket `mamba-ssm` versi **2.3.1** (rilis 10 Mar 2026) mengekspos `Mamba2` dengan API sederhana: menerima tensor `[batch, length, dim]` dan mengembalikan shape yang sama; parameter utama: `d_model`, `d_state` (umumnya 64 atau 128), `d_conv`, `expand`. citeturn14view0turn28search3  
Paper Mamba dan Mamba-2 menekankan SSM sebagai model sekuens linear-time; Mamba-2 (berbasis SSD) diklaim **2–8× lebih cepat** dari selective SSM Mamba awal, sehingga layak dipilih bila Anda ingin efisiensi tanpa mengganti keseluruhan backbone. citeturn28search0turn28search1  

### Desain blok pengganti yang paling “drop-in”: `DiTBlock` tetap, `Attention` diganti “MambaMixer”

Targetnya: perubahan sesedikit mungkin pada `DiTBlock`. Di kode, bagian yang diganti hanya baris:

`attn_output = self.attn(x=norm, mask=mask, rope=rope)` citeturn13view0  

menjadi:

`mix_output = self.mamba(norm, mask=mask)`  

lalu tetap: `x = x + gate_msa * mix_output`.

Skema ini menjaga:

- Input modulasi (`norm`) berasal dari `AdaLayerNorm` yang sama. citeturn13view0turn12view0  
- Output tetap masuk residual bergate yang sama. citeturn13view0  
- Jalur MLP dan modulasi MLP (`shift_mlp/scale_mlp/gate_mlp`) tetap identik. citeturn13view0  

### Handling positional & masking tanpa mengubah “kontrak” F5-TTS

1) **Posisional**  
F5-TTS sudah menambahkan *convolutional position embedding* pada output `InputEmbedding`: `x = conv_pos_embed(x, mask) + x`. citeturn9view0turn30view2  
Selain itu, attention memakai **RoPE** (RotaryEmbedding) yang dikirim sebagai `rope` ke blok-blok. citeturn10view4turn30view3  

Untuk integrasi konservatif:  
- Biarkan `ConvPositionEmbedding` tetap menjadi sumber positional utama.  
- Pada blok yang diganti, `rope` dapat **diabaikan** (tetap diterima di signature agar kompatibel). Ini meminimalkan perubahan; layer lain yang masih attention tetap memanfaatkan RoPE.

2) **Masking/padding**  
Mask dipakai di banyak tempat (mis. pada `ConvPositionEmbedding` untuk meng-*zero out* padding setelah conv) citeturn30view2 dan dibuat di `CFM.forward` dari `lens_to_mask`. citeturn29view0  

Untuk Mamba, paling aman:

- Jika `mask` ada: lakukan `x = x.masked_fill(~mask[...,None], 0)` sebelum Mamba dan juga `y = y.masked_fill(~mask[...,None], 0)` setelah. Ini meniru gaya masking conv/attention yang juga melakukan `masked_fill` saat mask tersedia. citeturn30view2turn30view3  
- Bila Anda memakai **BiMamba** (forward+reverse), lakukan *reverse by valid length* (bukan reverse tensor penuh) agar padding tidak menjadi “prefix” yang memengaruhi dinamika kausal pada arah balik.

### LayerNorm, residual, dan stabilitas numerik

Anda **tidak** memakai `Block` wrapper bawaan Mamba (yang strukturnya “Add → LN → Mixer” dan mengembalikan `(hidden_states, residual)` demi fusi kernel). Itu akan mengubah struktur blok F5-TTS. Kita sengaja mempertahankan struktur `DiTBlock` (AdaLN + gating) sehingga arsitektur tetap dekat dengan checkpoint. citeturn35view0turn13view0  

Stabilitas numerik penting karena dokumentasi `mamba-ssm` memperingatkan SSM sensitif terhadap presisi; mereka menyarankan mencoba penyimpanan parameter fp32 (mis. AMP default PyTorch) sebagai langkah pertama bila instabilitas muncul. citeturn14view0  

### Inisialisasi parameter yang “tidak merusak checkpoint”

Karena Anda melanjutkan dari checkpoint Indo, prinsipnya:

- **Semua parameter eksisting** (termasuk `attn_norm`, `ff_norm`, `ff`, embed, ConvNeXt, dsb.) dimuat dari checkpoint.  
- **Parameter baru Mamba** diinisialisasi default `mamba-ssm`.  
- Tambahkan **skalar pengali** `mamba_scale` atau *mixing gate* `alpha` yang *diinit* sehingga perilaku awal **menyamai model teacher**.

Pola paling konservatif:

- `alpha` scalar per-blok: output bloc = `(1-α) * attn_out + α * mamba_out`, dengan α mulai dari 0.  
- Saat α=0, output identik checkpoint (asumsi attention masih ada dan dimuat), sehingga Anda bisa mulai training tanpa “shock”.  
- Setelah Mamba mendekati attention (via distilasi), naikkan α → 1 dan *skip compute attention* saat inference.

## Distilasi hemat waktu untuk meminimalkan training tambahan

Fokus distilasi Anda adalah: **latih hanya modul baru** (Mamba + gating scalars) dan memanfaatkan checkpoint Indo sebagai teacher.

### Loss function yang disarankan (urut dari paling hemat waktu)

1) **Distilasi lokal per-blok (paling hemat)**  
Target: `mamba_out ≈ attn_out` pada blok yang diganti, di atas input `norm` yang sama (keluaran AdaLN). Karena attention sudah ada di blok itu, Anda bisa hitung `attn_out` dan `mamba_out` dalam satu forward, tanpa menjalankan teacher model penuh.

- `L_block = MSE(mamba_out, attn_out)` pada token/frame valid (mask).  
- Opsional: *Huber loss* bila outlier muncul (lebih stabil).  
Justifikasi: ini “mengunci” fungsi mixer baru agar meniru mixer lama, sehingga jumlah update yang dibutuhkan jauh lebih kecil dibanding fine-tune end-to-end.

2) **KD pada output flow (global)**  
CFM melatih `pred` untuk menebak `flow = x1 - x0` dan loss MSE di span mask. citeturn29view0turn21view0  
Tambahkan teacher-student:

- `L_kd = MSE(pred_student, pred_teacher)` pada span mask (atau semua frame).  
Ini membantu ketika Mamba tidak 100% mampu meniru attention secara lokal tetapi masih perlu menjaga fungsi keseluruhan.

3) **Loss asli CFM (ground truth)**  
- `L_cfm` persis seperti training asli: `MSE(pred, flow)` pada `rand_span_mask`. citeturn21view0turn29view0  

**Kombinasi konservatif**:  
`L_total = λ_block * L_block + λ_kd * L_kd + (1-λ_kd) * L_cfm`  
dengan `λ_block` tinggi di awal, lalu diturunkan setelah α mendekati 1.

### Schedule teacher–student yang meminimalkan waktu

Gunakan 3 tahap (semua jumlah update berupa variabel):

- Tahap A (warm-up distilasi lokal): α=0, latih Mamba untuk meniru attention pada 1 blok (`L_block` dominan). Output model tetap “teacher-like”.  
- Tahap B (progressive replacement): naikkan α dari 0 → 1 (mis. sigmoid/linear). Tambahkan `L_kd` dan sedikit `L_cfm` untuk memastikan output flow tetap valid.  
- Tahap C (stabilisasi akhir): α=1, matikan attention compute untuk blok itu, lanjutkan beberapa ribu update dengan `L_cfm` + kecil `L_kd` untuk fine alignment.

**Temperature** (interpretasi yang praktis untuk regresi):  
- Temperature klasik untuk distilasi softmax tidak langsung relevan. Untuk memenuhi kebutuhan “temperature”, gunakan **temperature pada fungsi transisi α** (mis. `α = sigmoid((step - s0)/τ)`; τ berperan sebagai “temperature” yang menghaluskan transisi). Ini sering lebih bermakna pada distilasi regresi dibanding “softening logits”.

### Data selection dari data Indo V2 untuk training singkat

Repo Indo V2 menyebut dataset finetune total **131.79 jam** dan **66,431 sampel** dari video YouTube Indonesia. citeturn38view0turn15view0  
Untuk minim waktu:

- Pilih subset berdurasi total `DATA_SUBSET_HOURS ∈ {2, 5, 10, 20}` jam untuk Tahap A/B; Tahap C bisa lebih kecil (mis. 1–5 jam).  
- Sampling sebaiknya mencakup variasi: gender/speaker, noise (bersih vs noisy), gaya formal vs kasual (sesuai contoh model card). citeturn38view0turn15view0  

### Progressive freezing (kunci hemat waktu)

Prioritas: hanya update parameter yang baru/krusial.

- Tahap A: trainable = `Mamba params` + `alpha/mamba_scale` (+ opsional `output_linear` kecil jika Anda pakai concat). Semua parameter lain dibekukan.  
- Tahap B: jika loss stagnan, buka trainable tambahan yang paling dekat: `attn_norm` dan/atau `ff_norm` di blok yang diganti (bukan seluruh model).  
- Tahap C: tetap sempit; jangan membuka ConvNeXt/text embed kecuali ada masalah pengucapan serius.

## Protokol training dan checkpointing

Bagian ini memetakan protokol yang konsisten dengan paper + implementasi trainer resmi, tetapi disesuaikan agar **konservatif**.

### Hyperparameter yang direkomendasikan (berbasis praktik F5-TTS)

Anchor dari paper base training: AdamW, peak LR `7.5e-5`, warmup `20K updates`, linear decay, grad clip 1. citeturn37view0  
Trainer resmi juga menggunakan **AdamW fused** dan scheduler **LinearLR warmup + LinearLR decay** (SequentialLR). citeturn23view0turn23view2  

Untuk fine-tune sempit (hanya Mamba) biasanya Anda bisa menaikkan LR, tetapi demi konservatif:

- Optimizer: AdamW (fused jika tersedia), weight decay kecil (mis. 0.01) — *variabel*.  
- LR: mulai `1e-4` untuk parameter Mamba (lebih tinggi dari base LR karena param sedikit), atau `7.5e-5` bila ingin sangat aman.  
- Warmup: `WARMUP_UPDATES = min(500, 0.1 * total_updates)` (karena training pendek).  
- Grad clip: 1.0 (konsisten). citeturn37view0turn23view2  
- Mixed precision: fp16/bf16 untuk compute, tetapi pertimbangkan **fp32 untuk sebagian parameter Mamba** bila instabil (sesuai catatan mamba-ssm soal sensitivitas). citeturn14view0  

Batching di F5-TTS base config menggunakan *frame-wise batching* (`batch_size_type: frame`) dengan `batch_size_per_gpu` besar, dan base total batch 307,200 frames pada 8 GPU. citeturn18view0turn37view0  
Untuk Anda: set `BATCH_FRAMES_PER_GPU` sesuai VRAM, dan batasi `max_samples` agar tidak OOM.

### Melanjutkan dari checkpoint Eempostor Indo V2

Repo model menyediakan file utama `f5_tts_indo_v2.pt` dan `vocab.txt`. citeturn38view1turn26view0  
Loader resmi mendukung `.pt` maupun `.safetensors`, dan bisa memuat EMA dengan `use_ema=True` (bila checkpoint berisi `ema_model_state_dict`) atau non-EMA (`use_ema=False`). citeturn26view2turn38view1  

Praktik konservatif:

- Untuk distilasi/training, gunakan bobot **non-EMA** sebagai student (karena EMA biasanya untuk inference), tetapi jadikan EMA (jika tersedia) sebagai teacher atau evaluasi. Paper menyebut inference memakai EMA. citeturn37view0turn26view2  
- Jika checkpoint Indo tidak menyertakan EMA, gunakan `use_ema=False` dan lanjutkan.

### Checkpointing & early stopping

Trainer resmi menyimpan checkpoint berkala (`save_per_updates`, `last_per_updates`) dan menyertakan `model_state_dict`, `ema_model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, dan `update`. citeturn27view1turn27view2  
Namun untuk eksperimen cepat:

- Simpan “last checkpoint” setiap `1k–2k updates`.  
- Simpan “milestone checkpoint” setiap `5k updates` atau saat metrik evaluasi membaik.  
- Early stop bila:  
  - `L_block` tidak turun >ε selama `patience` updates, atau  
  - UTMOS turun signifikan dan tidak pulih, atau  
  - WER naik signifikan (indikasi intelligibility drop).  

## Evaluasi ringan dan desain ablation

### Paket evaluasi yang sudah ada di repo F5-TTS (cepat dipakai)

Folder `src/f5_tts/eval` menyediakan workflow evaluasi objektif dan menyebut menjalankan **WER / SIM / UTMOS** pada hasil batch inference. citeturn33view0turn37view2  
- WER dan SIM-o juga dipakai di paper; paper menyebut WER memakai Whisper-large-v3 (untuk EN) dan SIM memakai model speaker verification berbasis WavLM-large. citeturn37view2  
- UTMOS bisa dijalankan via `eval_utmos.py`. citeturn33view0  

Untuk Indo: Anda bisa tetap menggunakan kerangka yang sama, hanya perlu memastikan ASR yang dipakai mendukung Bahasa Indonesia (Whisper mendukung multi-bahasa; paper sudah memakai whisper untuk EN). citeturn37view2  

### Tambahan metrik objektif yang ringan (MCD, F0 RMSE, ASR WER Indo)

1) **MOS proxy**  
- UTMOS adalah sistem prediksi MOS dari VoiceMOS Challenge 2022. citeturn31search21turn31search9  
- Alternatif cepat: SpeechMOS menyediakan MOS predictor via `torch.hub`. citeturn31search22turn31search2  

2) **MCD (Mel-Cepstral Distortion)**  
MCD umum dipakai untuk mengukur perbedaan sekuens mel-cepstra; biasanya lebih kecil = lebih mirip. citeturn31search8turn31search4  
Anda dapat menghitung MCD pada pasangan (audio sintetis, audio referensi) dengan DTW alignment bila diperlukan.

3) **F0 RMSE**  
Gunakan ekstraktor F0 dari WORLD (Harvest/DIO). Harvest didesain untuk estimasi F0 reliabel dan dipakai luas. citeturn31search19turn31search31turn31search27  
RMSE F0 dihitung pada frame voiced (mask voiced) agar tidak bias.

4) **ASR WER**  
Konsisten dengan paper: WER dihitung dari transkrip ASR atas audio hasil sintesis dibanding teks target. citeturn37view2  

### Desain uji subjektif cepat (tanpa MOS panel besar)

Karena Anda ingin evaluasi ringan, lakukan *A/B preference* internal:

- 20–30 kalimat, mencakup: angka, singkatan, kata serapan EN, gaya formal & kasual.  
- 3 kondisi: Baseline Indo V2 (teacher), Hybrid-1block, Hybrid-2block.  
- 5–10 pendengar, headphone, random order, blind, pertanyaan: *naturalness*, *speaker similarity*, *pronunciation*.  
Output: persentase preferensi dan catatan error yang sering.

### Ablation yang wajib untuk memastikan “konservatif”

| Ablation | Variasi | Tujuan | Ukuran cepat |
|---|---|---|---|
| Jumlah blok diganti | 0 vs 1 vs 2 | Mengukur dampak kumulatif | WER, SIM, UTMOS citeturn33view0turn37view2 |
| Jenis Mamba | Causal vs BiMamba | Uji kebutuhan konteks kanan untuk infilling | WER↑/UTMOS↓ → indikasi masalah konteks |
| Distilasi | none vs `L_block` vs `L_block+L_kd` | Cari training tercepat yang stabil | Konvergensi `L_block`, WER |
| Kapasitas Mamba2 | `expand=1` vs `expand=2` | Tradeoff param/quality | UTMOS & WER |

## Risiko, mitigasi, estimasi compute/time, dan resep implementasi reproduksibel

### Failure modes yang paling mungkin dan mitigasinya

1) **Degradasi infilling (butuh konteks kanan)**  
Risiko meningkat bila Mamba kausal dipakai. Mitigasi: pakai **BiMamba2**, atau batasi penggantian 1 blok dan pakai distilasi kuat + α fade-in lambat. (Attention F5-TTS sendiri non-causal.) citeturn30view3turn13view0  

2) **Training instabil / loss NaN**  
SSM bisa sensitif terhadap presisi; mitigasi: gunakan AMP dengan parameter utama fp32 (atau minimal untuk parameter Mamba), turunkan LR, naikkan warmup, gunakan grad clip 1. citeturn14view0turn37view0  

3) **Mismatch checkpoint loading**  
Karena mengganti modul, state dict untuk attention tidak 1–1. Mitigasi: gunakan mode “dual-branch” (attention tetap ada) sehingga load checkpoint tetap penuh, dan Mamba ditambahkan sebagai modul baru; hanya param Mamba yang missing (diinit). File Indo menyediakan `f5_tts_indo_v2.pt` dan loader resmi sudah menangani `.pt` serta opsi EMA. citeturn38view1turn26view2  

4) **Overfitting pada subset Indo**  
Mitigasi: sampling subset beragam (noise/clean, formal/kasual), gunakan early stopping berbasis UTMOS+WER. citeturn38view0turn33view0  

### Estimasi parameter & compute yang relevan untuk “konservatif”

`Mamba2` pada PyPI menyatakan parameter kira-kira `~ 3 * expand * d_model^2`. citeturn14view0  
Dengan `d_model = 1024`:

- `expand=1` → ~3.15M parameter per Mamba2.  
- `expand=2` → ~6.29M parameter per Mamba2.  
- BiMamba2 (2 arah) mengalikan ~2×.

Karena Anda hanya mengganti 1–2 blok dari total 22, **pertambahan parameter total** tetap kecil dibanding base model 335.8M. citeturn37view0turn14view0  

### Estimasi waktu training (range konservatif + cara menghitung)

Anchor paper: base training 1.2M updates “lebih dari seminggu” pada 8×A100 80GB, batch 307,200 frames; sehingga *order of magnitude* throughput global sekitar beberapa update/detik. citeturn37view0  

Karena eksperimen Anda:
- hanya 1–2 blok berubah,
- Anda akan melakukan training jauh lebih sedikit update,
- distilasi lokal menambah compute hanya pada beberapa blok (bukan teacher forward penuh),

maka estimasi waktu lebih tepat disajikan sebagai:

`TIME ≈ (N_UPDATES_TOTAL) * (t_update_baseline + t_overhead_distill)`  

dan Anda seharusnya mengukur `t_update_baseline` dengan 200 update awal.

Tabel tradeoff konservatif (isi `t_update` dan GPU Anda sendiri saat eksekusi):

| Run | `BLOCK_IDS` | Distilasi | Overhead forward | `N_UPDATES_TOTAL` (contoh) | Perkiraan durasi |
|---|---:|---|---:|---:|---|
| R0 (baseline) | 0 | none | 1× | 0 | 0 |
| R1 (paling konservatif) | 1 blok | `L_block` saja | ~1× + Mamba(1 blok) | 5k–15k | pendek–sedang |
| R2 (lebih stabil) | 1 blok | `L_block + L_kd` | ~1× + Mamba + KD head | 10k–25k | sedang |
| R3 (2 blok) | 2 blok | `L_block + L_kd` | ~1× + Mamba(2 blok) | 20k–40k | sedang–panjang |

Catatan: Bila Anda memakai teacher model penuh untuk `L_kd`, overhead bisa ~2× forward (lebih mahal). Karena itu saya rekomendasikan “self-distillation lokal” lewat dual-branch per blok.

### Diagram arsitektur yang disarankan (Mermaid)

```mermaid
flowchart LR
  subgraph Input
    A[Ref audio -> MelSpec] --> X1[mel x1]
    T[Text -> tokenizer/vocab] --> TE[TextEmbedding]
  end

  subgraph CFM_Training
    X0[Gaussian x0] --> PHI[phi_t = (1-t)x0 + t x1]
    TIME[t ~ U(0,1)] --> Temb[TimestepEmbedding]
    X1 --> MASK[Random span mask -> cond]
    MASK --> COND[cond audio (masked)]
    PHI --> INP
    COND --> INP
    TE --> INP
  end

  INP[InputEmbedding + ConvPositionEmbedding] --> DIT[DiT backbone (22 blocks)]
  Temb --> DIT
  DIT --> OUT[Proj -> predicted flow]
  OUT --> LOSS[MSE on masked span]

  subgraph Hybrid_Block
    direction TB
    B0[DiTBlock k] -->|keep AdaLN + FFN| B1
    B1[Replace Attention -> Mamba2/BiMamba2]
  end

  DIT -. blocks k in BLOCK_IDS .- Hybrid_Block
```

### Timeline training konservatif (Mermaid Gantt)

```mermaid
gantt
  title Timeline eksperimen konservatif (contoh)
  dateFormat  YYYY-MM-DD
  axisFormat  %d

  section Setup
  Implement patch & load ckpt            :a1, 2026-04-10, 1d
  Sanity check parity (alpha=0)          :a2, after a1, 1d

  section Distilasi lokal
  Stage A: L_block, alpha=0              :b1, after a2, 2d
  Stage B: alpha ramp + L_kd + L_cfm     :b2, after b1, 2d
  Stage C: alpha=1, fine-tune singkat    :b3, after b2, 1d

  section Evaluasi
  Batch infer + WER/SIM/UTMOS            :c1, after b3, 1d
  Ablation & quick listening test        :c2, after c1, 1d
```

### Resep implementasi langkah demi langkah + perintah (pseudocode gaya PyTorch)

Di bawah ini resep yang *reproducible* namun tetap mengandung variabel yang harus Anda set (`BLOCK_IDS`, path data, dsb.). Semua path adalah contoh.

#### Setup environment

```bash
# 1) Clone repo F5-TTS
git clone https://github.com/SWivid/F5-TTS.git
cd F5-TTS

# 2) Install F5-TTS (editable) dan dependency evaluasi bila diperlukan
pip install -e ".[eval]"   # untuk WER/SIM/UTMOS pipeline repo citeturn33view0

# 3) Install Mamba SSM v2.3.1 (GPU Linux + CUDA)
pip install "mamba-ssm==2.3.1" "causal-conv1d>=1.4.0"  # optional extra disebut di PyPI citeturn14view0
```

#### Implementasi modul `BiMamba2` dan blok hybrid

Buat file baru `src/f5_tts/model/mamba_patch.py` (contoh pseudocode):

```python
import torch
import torch.nn as nn
from mamba_ssm import Mamba2  # API usage di PyPI citeturn14view0

def reverse_by_length(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    # x: [B, N, D], reverse hanya prefix valid [0:length)
    B, N, D = x.shape
    out = x.clone()
    for b in range(B):
        L = int(lengths[b].item())
        if L > 1:
            out[b, :L] = torch.flip(x[b, :L], dims=[0])
        if L < N:
            out[b, L:] = 0.0
    return out

class BiMamba2(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 1):
        super().__init__()
        self.fwd = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.out_scale = nn.Parameter(torch.tensor(0.0))  # start at 0 -> safe (konservatif)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, N, D]
        if mask is None:
            y_f = self.fwd(x)
            y_b = self.bwd(torch.flip(x, dims=[1]))
            y_b = torch.flip(y_b, dims=[1])
        else:
            lengths = mask.sum(dim=1)
            x0 = x.masked_fill(~mask[..., None], 0.0)
            y_f = self.fwd(x0)
            x_rev = reverse_by_length(x0, lengths)
            y_b = self.bwd(x_rev)
            y_b = reverse_by_length(y_b, lengths)  # reverse back
            y_f = y_f.masked_fill(~mask[..., None], 0.0)
            y_b = y_b.masked_fill(~mask[..., None], 0.0)

        y = (y_f + y_b) * 0.5
        return self.out_scale * y  # start 0, kemudian belajar naik
```

Lalu buat “attention+mamba mixer” untuk distilasi lokal (dual branch):

```python
class AttnMambaMixer(nn.Module):
    def __init__(self, attn_module: nn.Module, mamba_module: nn.Module):
        super().__init__()
        self.attn = attn_module
        self.mamba = mamba_module
        self.alpha = nn.Parameter(torch.tensor(0.0))  # sigmoid(alpha)->0 at start

    def forward(self, x_norm, mask=None, rope=None, return_branches=False):
        attn_out = self.attn(x=x_norm, mask=mask, rope=rope)  # original path citeturn13view0turn30view3
        mamba_out = self.mamba(x_norm, mask=mask)

        a = torch.sigmoid(self.alpha)
        out = (1 - a) * attn_out + a * mamba_out

        if return_branches:
            return out, attn_out, mamba_out, a
        return out
```

#### Patch blok DiT yang dipilih

```python
from f5_tts.model.backbones.dit import DiT
from f5_tts.infer.utils_infer import load_checkpoint  # loader ckpt citeturn26view2
from f5_tts.model.mamba_patch import BiMamba2, AttnMambaMixer

BLOCK_IDS = [11]  # <== set variabel Anda

dit = DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4, pe_attn_head=1,
          attn_backend="torch", attn_mask_enabled=False)  # sesuai config base citeturn18view0turn37view0

# Load weights dari Indo checkpoint
ckpt_path = "checkpoints/f5_tts_indo_v2.pt"  # file dari repo Eempostor citeturn38view1
dit = load_checkpoint(dit, ckpt_path, device="cuda", use_ema=False)

# Patch 1–2 blok
for k in BLOCK_IDS:
    block = dit.transformer_blocks[k]
    old_attn = block.attn
    mamba = BiMamba2(d_model=dit.dim, d_state=64, d_conv=4, expand=1)
    block.attn = AttnMambaMixer(old_attn, mamba)
    # Freeze attention teacher
    for p in block.attn.attn.parameters():
        p.requires_grad = False
```

#### Training loop konservatif (distilasi lokal + CFM) — pseudocode

Anda bisa membungkus ini di script `train_mamba_distill.py`:

```python
# Pseudocode: gunakan CFM.forward untuk batch training loss asli citeturn29view0turn21view0
# Tambahkan L_block dari setiap AttnMambaMixer

lambda_block = 1.0
lambda_cfm = 0.1

for batch in loader:
    loss_cfm, cond, pred = cfm(mel, text=text, lens=mel_lengths)
    loss_block_total = 0.0

    # Hook sederhana: jalankan forward kedua? (hemat: modifikasi DiTBlock agar return_branches)
    # Cara praktis: patch DiTBlock.forward untuk call block.attn(... return_branches=True) jika mixer bertipe AttnMambaMixer

    for k in BLOCK_IDS:
        # ambil attn_out & mamba_out dari cache yang Anda simpan di forward
        loss_block_total += mse(mamba_out_k, attn_out_k)

    loss = lambda_cfm * loss_cfm + lambda_block * loss_block_total
    loss.backward()
    opt.step(); sched.step(); opt.zero_grad()
```

#### Perintah menjalankan training (template)

Repo F5-TTS menggunakan `accelerate launch` untuk training. citeturn34view0  
Contoh template (Anda menambahkan script Anda sendiri):

```bash
accelerate config  # sekali citeturn34view0turn33view0

accelerate launch --mixed_precision=fp16 \
  src/f5_tts/train/train_mamba_distill.py \
  --ckpt_path checkpoints/f5_tts_indo_v2.pt \
  --vocab_file checkpoints/vocab.txt \
  --block_ids "11" \
  --batch_frames_per_gpu 16000 \
  --total_updates 15000 \
  --warmup_updates 500 \
  --lr 1e-4
```

### Template chart untuk membandingkan metrik (kode plotting)

Setelah Anda punya file hasil evaluasi (mis. JSONL dari eval scripts), buat plot:

```python
import json, glob
import matplotlib.pyplot as plt

runs = {
  "baseline": "results/baseline_metrics.json",
  "hybrid_1blk": "results/hybrid1_metrics.json",
  "hybrid_2blk": "results/hybrid2_metrics.json",
}

metrics = ["wer", "sim", "utmos", "mcd", "f0_rmse"]
vals = {m: [] for m in metrics}
labels = []

for name, path in runs.items():
    with open(path) as f:
        d = json.load(f)
    labels.append(name)
    for m in metrics:
        vals[m].append(d.get(m, None))

for m in metrics:
    plt.figure()
    plt.bar(labels, vals[m])
    plt.title(m)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()
```

## Tabel ringkas distilasi vs compute (pilihan strategi)

| Strategi | Teacher | Loss | Compute overhead | Kapan dipakai |
|---|---|---|---|---|
| Distilasi lokal dual-branch (disarankan) | Attention blok yang sama (frozen) | `L_block` | Rendah (hanya 1–2 blok ekstra) | Paling cepat untuk “swap” konservatif |
| KD output flow (teacher full model) | Model teacher full | `L_kd` | Tinggi (~+1 forward full) | Jika distilasi lokal tidak cukup |
| Fine-tune CFM saja | Tidak ada | `L_cfm` | Normal | Jika Anda yakin swap kecil, tapi umumnya butuh steps lebih banyak citeturn21view0turn29view0 |

---

Semua rekomendasi di atas menargetkan prinsip konservatif: **jaga struktur `DiTBlock` (AdaLN + gating + FFN) apa adanya** citeturn13view0turn12view0, lakukan penggantian hanya pada komponen “mixer/attention”, manfaatkan `mamba-ssm` **v2.3.1** yang menyediakan `Mamba2` drop-in `[B,N,D]→[B,N,D]` citeturn14view0, dan gunakan distilasi lokal agar jumlah update tetap kecil sambil mempertahankan perilaku checkpoint Indo Anda (Eempostor). citeturn38view1turn26view2