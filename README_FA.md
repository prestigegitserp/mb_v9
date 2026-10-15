# MBMF V9 — بستهٔ اجرای مقاله با کنترل Validation و نمایش پیشرفت

> وضعیت: نسخهٔ پژوهشی **Beta**. تست‌های مستقل انتخاب ابرپارامتر اجرا شده‌اند؛ **آزمایش‌های سنگین واقعی و مقایسهٔ عددی V8/V9 اجرا نشده‌اند**. برای اجرای واقعی ابتدا هستهٔ تأییدشدهٔ V8 باید کپی شود.

## ۰. اصل غیرقابل‌تغییر

- مبنا: شاخهٔ `fix/windows-crlf-core-hash` از مخزن `prestigegitserp/mb_v8`، commit `f58ec7b95ddc802561cbc1399bcc17be226b10de`.
- هستهٔ علمی `mbmf_q1_final_v7.py` ثابت، SHA-256 استانداردشده `6da726cb155ab6ad291f5f608febc37f42d1c25ac3924910f56917908c1f124c`.
- ساخت V9 تنها به **خواندن/کپی** از V8 نیاز دارد؛ هیچ Git commit، stash، branch یا ویرایشی در V8 انجام نمی‌شود.
- V9 معماری، loss، trainer اصلی و فرمول‌های علمی روش‌های ۱ تا ۷ را تغییر نمی‌دهد.

## ۱. نصب ویندوز

در پوشهٔ جدید و مستقل `mb_v9`:

```powershell
# مسیر سورس باید clone کامل و بدون ویرایش commit فیکس باشد.
.\SETUP_WINDOWS.ps1 -V8Source "C:\path\to\mb_v8"
```

اگر V8 هنوز clone نشده:

```powershell
git clone --branch fix/windows-crlf-core-hash --single-branch https://github.com/prestigegitserp/mb_v8.git "C:\path\to\mb_v8"
```

سپس دیتاست‌ها را با `jupyter notebook vendor_v8/download_datasets_q1_v8.ipynb` از پوشهٔ V9 دانلود کنید؛ یا همان فایل‌های دیتاست قبلی خود را به `./datasets` بدهید.

## ۲. اجرای بسیار ساده

```powershell
# فقط برنامه و شمارش آموزش‌ها، بدون هزینهٔ آموزش
python run_v9.py plan

# اجرای همهٔ مراحل روی شش دیتاست؛ بسیار سنگین
.\RUN_EVERYTHING_WINDOWS.ps1

# ادامه بعد از قطع شدن: jobهایی که کامل‌اند دوباره اجرا نمی‌شوند
.\RUN_EVERYTHING_WINDOWS.ps1 -Resume

# یک دیتاست؛ تمام مراحل و پنج K
.\RUN_EVERYTHING_WINDOWS.ps1 -Datasets 'ml-100k' -Resume

# فقط تست کوتاه: این خروجی مقاله نیست
python run_v9.py main --datasets ml-100k --quick
```

مراحل `all` به ترتیب: `scan` (فقط Validation)، `final` (تثبیت هر روش با Validation و سپس Test)، `calibration` (Method7 با انتخاب Validation NLL جدا از مدل accuracy)، `ablation` (۳۸ نسخه با Test قفل‌شدهٔ خود)، `ablation-fixed` (تمام ۳۸ Ablation در K یکسان، برای کنترل مکانیزمی)، `kstudy` (Test توصیفی برای Kهای ثابت)، `sparsity` (حذف بخشی از TRAIN با Validation/Test ثابت)، `density` (تراکم Synthetic در یک آزمایش مستقل)، سپس Audit. تحلیل Temporal سه MovieLens هم با `-IncludeTemporal` اجرا می‌شود. مطالعهٔ مستقل Train/Validation RMSE در نقاط Epoch با `-IncludeConvergence` (هزینه‌بر) فعال می‌شود.

سایر فرمان‌ها:

```powershell
python run_v9.py self-test
python run_v9.py scan --datasets ml-1m --resume
python run_v9.py final --datasets ml-1m --resume
python run_v9.py calibration --datasets ml-1m --resume
python run_v9.py ablation --datasets ml-1m --resume
python run_v9.py ablation-fixed --datasets ml-1m --ablation-fixed-k 20 --resume
python run_v9.py kstudy --datasets ml-1m --resume
python run_v9.py sparsity --datasets ml-1m --resume
python run_v9.py density --datasets synthetic --resume
python run_v9.py temporal --datasets ml-1m --resume
python run_v9.py convergence --datasets ml-100k --k 20 --split-seeds 2026 --model-seeds 11 --eval-every 1
python run_v9.py curves --datasets ml-1m
python run_v9.py audit --datasets ml-1m
```

## ۳. دیباگ روش‌شناسی و رفع ایرادهای V8

1. **اسکن واقعی → فریز → تست:** V8 فقط در هر job اسکنِ جدا بهترین K را انتخاب می‌کرد و سپس ابرپارامترها را برای اجرای اصلی به‌طور خودکار منتقل نمی‌کرد. V9 همهٔ کاندیدهای Validation را با کلید config مستقل جمع می‌کند، برای هر Outer Split، میانگین Validation RMSE را فقط روی Model Seedهای همان Split می‌گیرد و **یک ترکیب برنده برای هر روش و هر Outer Split** تثبیت می‌کند. سپس همان ترکیب را با seedهای مشخص در مسیر Locked Test ارزیابی می‌کند.
2. **Joint Search:** علاوه بر One-Factor، زوج‌های معنی‌دار (M1: alpha/beta، M2: fusion/consistency، M3: scale/L2، M4: logit shift/L2، M5: fusion/reconstruction، M6: temperature/reconstruction و M7: context/tau) و اسکن‌های تکمیلی learning rate، weight decay، اندازهٔ لایهٔ پنهان و چند initialization آزموده می‌شوند؛ حالت پیش‌فرض `hybrid` است. همچنان کل فضای دکارتی همهٔ پارامترها نیست؛ شمارش دقیق با `plan` نمایش داده می‌شود.
3. **رد Data-Protocol در Hyper Selection:** تغییر train fraction یا synthetic density در search به دلیل مقایسه‌ناپذیری splitها حذف شده است؛ اینها باید sensitivity study مستقل باشند. مشکل Train=.90 و Test=0 در V8 دیگر مسیر Hyperparameter Selection را متوقف نمی‌کند.
4. **عدم نشت بین Outer Splitها:** انتخاب ابرپارامتر هر Split منحصراً از Validation همان Split انجام می‌شود. میانگین‌گیری برای انتخاب برنده بین Splitهای تصادفی مختلف مجاز نیست، چون Test یک Split ممکن است در Validation Split دیگر وجود داشته باشد.
5. **هیچ Test Leakage:** اسکن همیشه `validation_only=True` است. اگر حتی یک ستون `test_` در جدول کاندید وارد شود، V9 خطا می‌دهد. انتخاب برنده فقط Validation RMSE دارد.
6. **پوشش مساوی Replicate:** کاندید ناقص یا با تعداد Split/Seed متفاوت نمی‌تواند برنده شود.
7. **Progress:** هر candidate با روش، K، initialization، seed و درصد / ETA ثبت می‌شود، همراه با heartbeat در کارهای طولانی و log اجرای هر مرحله.
8. **K Study:** Train/Validation/Test RMSE برای Kهای `5,10,20,50,64` جداگانه قابل مقایسه است؛ برای هر روش نمودار جدا ایجاد می‌شود؛ Test **برای انتخاب K نیست**.
9. **Epoch:** Validation RMSE در تیک‌های ارزیابی در `V9_EPOCH_METRICS.csv` ثبت می‌شود. با `--epoch-train-rmse`، Train RMSE نیز در همان تیک‌ها محاسبه می‌شود (هزینه‌بر). تاریخچهٔ MBMF Sparse همچنان Objective دارد و Validation RMSE در هر Epoch به علت عدم گزارش در Trainer اصلی در V9 بدون تغییر علمی تضمین نمی‌شود. Test در Epochها هرگز برای انتخاب استفاده نمی‌شود.
10. **Train-only sparsity:** قبل از مدل‌سازی، بخشی از ردیف‌های **TRAIN** حذف می‌شود و Val/Test اصلی ثابت می‌مانند؛ حداقل پوشش هر User/Item حفظ می‌شود. مقدار Keep در اثر الزام پوشش می‌تواند اندکی با مقدار درخواستی متفاوت باشد.
11. **محیط و هویت:** سورس علمی با SHA مشخص و خواندن Checkpoint با بررسی هویت همان کد صورت می‌گیرد.

## ۴. خروجی‌ها

```
results_v9/<dataset>/
  search/
    jobs/<scan-job>/candidate_validation_runs.csv
    hyperparameter_validation_all.csv
    validation_ranked_configs.csv
    validation_winners.csv
    validation_selection_manifest.json
  final/<method>/split_<split-seed>/locked_test_runs.csv
  calibration/calibration_locked_test_runs.csv # Method7 NLL-tuned only
  final_locked_test_all.csv
  final_locked_test_summary.csv
  ablations/<tag>/locked_test_runs.csv
  ablations_matched_K_20/<tag>/locked_test_runs.csv
  ablations/ablation_summary.csv
  kstudy/K_<k>/locked_test_runs.csv
  kstudy/figures/<dataset>_K_RMSE_train_val_test.png|pdf
  sparsity/keep_<rate>/locked_test_runs.csv
  synthetic_density/density_<rate>/locked_test_runs.csv
  temporal/locked_test_runs.csv
  convergence/V9_EPOCH_METRICS.csv # only train/validation metrics, never test
  v9_validation_audit.json
```

هر اجرای training فایل `V9_JOB_STATUS.json` دارد: `RUNNING / COMPLETED / FAILED`، Fingerprint تنظیمات و تعداد کاندیدها. فایل `V9_EPOCH_METRICS.csv` شامل تیک‌های Validation و در صورت درخواست Train است.

## ۵. ساخت **ریپازیتوری جدید** در GitHub

این نسخه به‌شکل بستهٔ محلی ارائه شده است؛ اتصال GitHub چت حاضر عمل «ایجاد مخزن تازه» ندارد. پس از اجرای Bootstrap:

```powershell
.\PUBLISH_NEW_REPO.ps1
```

این اسکریپت با `gh repo create prestigegitserp/mb_v9 --private` مخزن کاملاً جدید را می‌سازد و فقط **دایرکتوری V9** را push می‌کند. پیش از اجرا باید `gh auth login` کرده باشید. اگر `origin` به ریپوی دیگری اشاره کند، اسکریپت با خطا متوقف می‌شود. هیچ بخشی از V8 نوشته نمی‌شود.

## ۶. محدودیت‌های صادقانه

- تمام ترکیب‌های مستقل همهٔ Hyperparameterها آزمایش نمی‌شوند؛ `hybrid` مجموعهٔ شفاف و گسترده‌ای از One-Factor و زوج‌های معنی‌دار است.
- هزینهٔ محاسباتی بسیار بالاست. `python run_v9.py plan` را قبل از هر اجرا بررسی کنید.
- Code/Weights رقبا BMC-ADMM، Deep-MMMF و DRPRA همراه V8 وجود ندارند و نیازمند اجرای author-code هستند؛ V9 نتایج آنها را اختراع نمی‌کند.
- Validation انتخاب مدل را انجام می‌دهد؛ Testهای Ablation و K Study صرفاً گزارش/تحلیل مکانیزم و حساسیت‌اند، نه tuning.
- بدون اجرای واقعی روی دیتاست‌های کامل، صحت نتیجهٔ RMSE یا سرعت نهایی را نمی‌توان ادعا کرد.

جزئیات اشکال‌های مهم انتخاب بین Splitها، بازتولید، و محدودیت‌های ارزیابی در [ممیزی پژوهشی V9](docs/V9_RESEARCH_AUDIT_FA.md) مستند شده است.

پس از انتشار V9 در مخزن تازه، گردش‌کار `.github/workflows/unit-tests.yml` تست‌های واحد را روی push و PR اجرا می‌کند و اگر هستهٔ V8 در مخزن vendored باشد، Self-Test علمی را هم انجام می‌دهد. این CI جای آزمایش سنگین واقعی را نمی‌گیرد.
