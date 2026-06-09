// generate laporan_KOM1338.docx
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, BorderStyle, WidthType, ShadingType, LevelFormat, PageBreak
} = require('docx');
const fs = require('fs');
const path = require('path');

// ── Layout (A4, margins: top/right/bottom 2.54 cm, left 3 cm) ──────────────
const PAGE_W   = 11906;
const M_LEFT   = 1701; // 3 cm  (1440 * 3/2.54 ≈ 1701)
const M_OTHER  = 1440; // 2.54 cm
const CONT_W   = PAGE_W - M_LEFT - M_OTHER; // 8765 DXA
const COL1     = 6200;
const COL2     = CONT_W - COL1; // 2565

const TNR  = 'Times New Roman';
const MONO = 'Courier New';

// sizes (half-points)
const T_TITLE = 28; // 14pt
const T_HEAD  = 24; // 12pt
const T_BODY  = 22; // 11pt
const T_CODE  =  16; // 8pt

// ── spacing helpers ──────────────────────────────────────────────────────────
const sp = (before, after, line = 240) =>
  ({ before, after, line, lineRule: 'auto' });

// ── paragraph factories ──────────────────────────────────────────────────────
function para(runs_or_str, opts = {}) {
  const children = typeof runs_or_str === 'string'
    ? [new TextRun({ text: runs_or_str, font: TNR,
        size: opts.size || T_BODY, bold: !!opts.bold })]
    : runs_or_str;
  return new Paragraph({
    children,
    alignment: opts.align !== undefined ? opts.align : AlignmentType.JUSTIFIED,
    spacing: opts.spacing || sp(0, 80),
    pageBreakBefore: !!opts.newPage,
  });
}

function heading(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, font: TNR, size: T_HEAD, bold: true })],
    alignment: AlignmentType.LEFT,
    spacing: sp(180, 60),
    pageBreakBefore: !!opts.newPage,
  });
}

function subheading(text) {
  return new Paragraph({
    children: [new TextRun({ text, font: TNR, size: T_BODY, bold: true, italics: true })],
    alignment: AlignmentType.LEFT,
    spacing: sp(100, 40),
  });
}

function bul(runs_or_str) {
  const children = typeof runs_or_str === 'string'
    ? [new TextRun({ text: runs_or_str, font: TNR, size: T_BODY })]
    : runs_or_str;
  return new Paragraph({
    numbering: { reference: 'bullets', level: 0 },
    children,
    spacing: sp(0, 40),
  });
}

// ── table helpers ────────────────────────────────────────────────────────────
const brd = { style: BorderStyle.SINGLE, size: 4, color: '000000' };
const borders = { top: brd, bottom: brd, left: brd, right: brd };

function trow(texts, header = false) {
  return new TableRow({
    tableHeader: header,
    children: texts.map((txt, i) => new TableCell({
      borders,
      width: { size: i === 0 ? COL1 : COL2, type: WidthType.DXA },
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      shading: header ? { fill: 'CCCCCC', type: ShadingType.CLEAR } : undefined,
      children: [new Paragraph({
        children: [new TextRun({
          text: txt, font: TNR, size: T_BODY,
          bold: header,
        })],
        alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
        spacing: sp(0, 0, 240),
      })]
    }))
  });
}

// ── code helpers ─────────────────────────────────────────────────────────────
function cline(line) {
  return new Paragraph({
    children: [new TextRun({ text: line === '' ? ' ' : line, font: MONO, size: T_CODE })],
    spacing: { before: 0, after: 0, line: 180, lineRule: 'auto' },
  });
}

function codeBlock(src) {
  return src.split('\n').map(cline);
}

// ── read source files ─────────────────────────────────────────────────────────
const pipelineSrc = fs.readFileSync(path.join(__dirname, 'src', 'loan_pipeline.py'), 'utf8');
const trainSrc    = fs.readFileSync(path.join(__dirname, 'train_model.py'), 'utf8');

// ── runs helpers ──────────────────────────────────────────────────────────────
const run  = (t, bold) => new TextRun({ text: t, font: TNR, size: T_BODY, bold: !!bold });
const bold = (t) => run(t, true);

// ── document ──────────────────────────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [{
      reference: 'bullets',
      levels: [{
        level: 0,
        format: LevelFormat.BULLET,
        text: '-',
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 420, hanging: 240 } } }
      }]
    }]
  },
  sections: [{
    properties: {
      page: {
        size: { width: PAGE_W, height: 16838 },
        margin: { top: M_OTHER, right: M_OTHER, bottom: M_OTHER, left: M_LEFT }
      }
    },
    children: [

      // ── JUDUL ──────────────────────────────────────────────────────────
      para('LAPORAN KOM1338 DATA MINING', {
        size: T_TITLE, bold: true,
        align: AlignmentType.CENTER,
        spacing: sp(0, 20),
      }),
      para('Prediksi Loan Approval', {
        size: T_TITLE, bold: true,
        align: AlignmentType.CENTER,
        spacing: sp(0, 80),
      }),
      para('Nama  :  [Nama Mahasiswa]          NIM  :  [NIM]', {
        align: AlignmentType.CENTER,
        spacing: sp(0, 0),
      }),

      // Garis pemisah
      new Paragraph({
        children: [],
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: '000000', space: 1 } },
        spacing: sp(60, 140),
      }),

      // ── 1. DESKRIPSI MASALAH ───────────────────────────────────────────
      heading('1. Deskripsi Masalah'),
      para(
        'Kompetisi KOM1338 merupakan tugas klasifikasi biner yang bertujuan memprediksi ' +
        'kemungkinan kegagalan pembayaran kredit (loan_status: 0 = lunas, 1 = default) ' +
        'berdasarkan data peminjam dan pinjaman. Dataset terdiri atas 43.983 data latih dan ' +
        '14.662 data uji dengan 12 fitur yang mencakup informasi demografis peminjam (usia, ' +
        'pendapatan tahunan, lama bekerja), karakteristik pinjaman (jumlah, suku bunga, tujuan, ' +
        'peringkat kredit), dan riwayat kredit (panjang riwayat, status default sebelumnya). ' +
        'Dataset bersifat tidak seimbang dengan proporsi kelas positif (default) sebesar 14,2%. ' +
        'Evaluasi dilakukan menggunakan metrik AUC-ROC, sehingga prediksi yang dikumpulkan ' +
        'berupa probabilitas kontinu (0–1), bukan label biner.'
      ),

      // ── 2. METODE ──────────────────────────────────────────────────────
      heading('2. Metode yang Digunakan'),

      subheading('2.1  Pra-pemrosesan dan Rekayasa Fitur'),
      bul([bold('Ordinal encoding'), run(' pada loan_grade (A=1 hingga G=7): tingkat default naik monoton dari A=5% hingga G=85%, sehingga representasi bilangan bulat lebih tepat daripada one-hot.')]),
      bul([bold('One-hot encoding'), run(' pada person_home_ownership, loan_intent, dan cb_person_default_on_file untuk LightGBM/XGBoost; CatBoost menangani fitur kategorikal secara native (target statistics).')]),
      bul([bold('Fitur interaksi:'), run(' int_rate_x_grade = loan_int_rate × grade_ord. Fitur turunan lain (rasio pendapatan, flag anomali) diuji dan dihapus karena menurunkan AUC—tanda overfitting.')]),
      bul([bold('Tanpa class_weight/resampling:'), run(' AUC adalah metrik berbasis ranking; pembobotan kelas mendistorsi estimasi probabilitas dan terbukti menurunkan OOF AUC (0,9314 → 0,9319 tanpa bobot).')]),

      subheading('2.2  Model dan Validasi Silang'),
      bul('Tiga model gradient boosting: LightGBM, XGBoost, dan CatBoost. Ketiganya bersifat robust terhadap outlier, tidak memerlukan scaling, dan efisien pada data tabular.'),
      bul('Skema validasi: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)—menjaga proporsi kelas 14,2% di setiap fold, menghasilkan estimasi AUC yang tidak bias.'),
      bul([bold('Hyperparameter tuning'), run(' dengan Optuna (TPE sampler, 50 trial masing-masing untuk LightGBM dan XGBoost; CatBoost menggunakan parameter yang divalidasi secara manual). Parameter kunci: max_depth=4–5, learning_rate=0,01–0,03, regularisasi L1/L2.')]),
      bul([bold('Prediksi out-of-fold (OOF):'), run(' setiap baris data latih diprediksi oleh model yang tidak melihatnya saat pelatihan, sehingga AUC OOF merupakan estimasi generalisasi yang bebas kebocoran data.')]),

      subheading('2.3  Ensemble Rank-Blend'),
      bul('Prediksi tiap model dinormalisasi ke peringkat relatif (rank, 0–1) sebelum digabung—menjadikannya robust terhadap perbedaan skala dan kalibrasi antarmodel, dan selaras dengan sifat AUC sebagai metrik ranking.'),
      bul('Bobot optimal dicari melalui grid search integer pada OOF AUC. Bobot terbaik yang ditemukan: 3·LGB + 3·XGB + 1·Cat.'),
      bul('Prediksi akhir pada data uji dihitung sebagai rata-rata berbobot prediksi tiap fold, kemudian digabung dengan rank-blend yang sama.'),

      // ── 3. HASIL DAN KESIMPULAN ────────────────────────────────────────
      heading('3. Hasil dan Kesimpulan'),

      new Table({
        width: { size: CONT_W, type: WidthType.DXA },
        columnWidths: [COL1, COL2],
        rows: [
          trow(['Model', 'OOF AUC (5-Fold)'], true),
          trow(['LightGBM (50 trial Optuna)', '0,9324']),
          trow(['XGBoost (50 trial Optuna)', '0,9323']),
          trow(['CatBoost (parameter manual)', '0,9303']),
          trow(['Rank-Blend (3·LGB + 3·XGB + 1·Cat)', '0,9325 (terbaik)']),
        ]
      }),
      new Paragraph({ children: [], spacing: sp(80, 0) }),

      para(
        'Ensemble rank-blend menghasilkan AUC-ROC OOF terbaik sebesar 0,9325, melampaui ' +
        'target kompetisi (≥0,92). Temuan utama: (1) loan_grade adalah prediktor terkuat ' +
        'dengan pola monoton yang konsisten; (2) penambahan fitur rekayasa berlebih justru ' +
        'menurunkan performa akibat overfitting noise; (3) tidak menggunakan pembobotan kelas ' +
        'terbukti lebih baik untuk optimasi AUC; (4) ensemble tiga model gradient boosting ' +
        'yang error-nya terdekorelasi secara konsisten melampaui model tunggal terbaik. ' +
        'Keterbatasan utama adalah tidak digunakannya data eksternal demi menjaga reproducibility ' +
        'dan integritas akademik, sehingga ceiling AUC berada pada kisaran 0,932–0,935.'
      ),

      // ── LAMPIRAN ───────────────────────────────────────────────────────
      heading('Lampiran — Kode Program', { newPage: true }),
      para(
        'Berikut adalah kode program lengkap yang digunakan untuk menghasilkan submission terbaik. ' +
        'Modul src/loan_pipeline.py memuat seluruh logika inti (feature engineering, cross-validation, ' +
        'tuning, dan ensemble). Skrip train_model.py adalah orkestrator yang memanggil modul tersebut.',
        { spacing: sp(0, 120) }
      ),

      subheading('File: src/loan_pipeline.py'),
      ...codeBlock(pipelineSrc),

      new Paragraph({ children: [], spacing: sp(120, 0) }),
      subheading('File: train_model.py'),
      ...codeBlock(trainSrc),

    ]
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync('laporan_KOM1338.docx', buf);
  console.log('Wrote laporan_KOM1338.docx');
});
