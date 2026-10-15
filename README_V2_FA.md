# MBMF V2 — راهنمای شاخهٔ توسعه (Research Candidate)

> **مهم:** این بازطراحی بر اساس تحلیل واقعی ML-100K و 21 تست واحد شکل گرفته است، **اما بهبود RMSE هنوز با اجرای کامل دیتاست‌های واقعی اثبات نشده** است. اعداد قبلی به V1 مربوط‌اند.

## Branches و جداسازی نسخه‌ها

- `main`: نسخهٔ یک ثابت؛ هیچ تغییری در آن داده نشده است.
- `v2`: شاخهٔ مقصد نسخهٔ دو؛ **فعلاً همان Commit اولیهٔ main** و منتظر تأیید کاربر است.
- `v2-development`: تمام تغییرات V2، تست‌ها و مستندات در این شاخه آماده می‌شود؛ به `v2` مرج نشده است.

### Clone نسخهٔ یک
```powershell
git clone --branch main --single-branch https://github.com/prestigegitserp/mb_v9.git mb_v9_v1
```

### Clone نسخهٔ در حال توسعهٔ دو
```powershell
git clone --branch v2-development --single-branch https://github.com/prestigegitserp/mb_v9.git mb_v9_v2
cd mb_v9_v2
python -m pip install -r requirements.txt
python -m pytest -q tests/test_v2_research.py tests/test_v9_validation.py
python run_v9.py plan --datasets ml-100k --methods Method1,Method2,Method3,Method4,Method5,Method6,Method7 --k 20
```

> روی شاخهٔ V2، اسکریپت معمول `run_v9.py` از **هفت مدل بازطراحی‌شده** استفاده می‌کند. نام‌های Method1...Method7 تغییر نکرده؛ هفت Variant اضافی ایجاد نشده‌اند.

### مقایسهٔ ظرفیت: فقط Validation، نه Test

**هشدار:** فرمان `main` برای مدل‌های منتخب Test هم محاسبه می‌کند؛ بنابراین مقایسهٔ سه پروفایل با `main` برای انتخاب برنده، Test Leakage ایجاد می‌کند. در مرحلهٔ تنظیم فقط `scan` اجرا کن. انتخاب پروفایل را پیش از هر اجرای `final` و تنها با Validation انجام بده.

```powershell
python run_v9.py plan --datasets ml-100k --methods Method1,Method2,Method3,Method4,Method5,Method6,Method7 --k 20 --scan-mode one-factor
python run_v9.py scan --datasets ml-100k --data-root ./datasets --split-seeds 2026 --model-seeds 1,11,22 --k 20 --methods Method1,Method2,Method3,Method4,Method5,Method6,Method7 --scan-mode one-factor --v2-profile regularized --output ./results_v2/regularized --resume
python run_v9.py scan --datasets ml-100k --data-root ./datasets --split-seeds 2026 --model-seeds 1,11,22 --k 20 --methods Method1,Method2,Method3,Method4,Method5,Method6,Method7 --scan-mode one-factor --v2-profile balanced --output ./results_v2/balanced --resume
python run_v9.py scan --datasets ml-100k --data-root ./datasets --split-seeds 2026 --model-seeds 1,11,22 --k 20 --methods Method1,Method2,Method3,Method4,Method5,Method6,Method7 --scan-mode one-factor --v2-profile capacity --output ./results_v2/capacity --resume
```

جدول‌های `search/hyperparameter_validation_all.csv` این سه مطالعه را برای هر روش و هر Split مقایسه کن، سپس فقط پروفایل برندهٔ ازپیش‌تعیین‌شده با Validation را به `final` بده. انتخاب خودکار «بهترین پروفایل بین این سه اسکن» هنوز در V2 RC پیاده نشده است؛ تا قبل از اجرای آن نباید با دیدن Test تصمیم گرفت.

برای غربال اولیه `--quick` مجاز است؛ **نتایج آن سه Epoch دارند و برای مقاله معتبر نیستند**.

## تغییرات هفت روش

| مدل | تغییر اصلی | نقص هدف‌گیری‌شده |
|---|---|---|
| M1 | مرکز Bias شرطی، جهت‌های Cartesian ایزوتروپیک و Reconstruction کم‌وزن | گیرکردن در مرکز ۳، تعارض Loss و Overfitting |
| M2 | Fusion وابسته به تراکم همراه با Consistency واقعاً مستقل | تجزیهٔ جبری Loss قبلی و Fusion ضعیف |
| M3 | مرکز یادگرفتنی، Warm-up جهت‌ها، کف Gain و Anchor شعاع | فروریزش شعاع و خاموشی گرادیان زاویه |
| M4 | Residual نامتقارن دارای کران و تصحیح Logit | اشباع و همگرایی کند |
| M5 | شعاع مستقلِ سمت مقابل با Decoder فقط به‌عنوان Prior، Loss پیش‌بینی‌محور | شکست Decoder-norm coupling، غلبهٔ Reconstruction |
| M6 | نسخهٔ Range-safe با دمای وابسته به K و شعاع یادگرفتنی | خروجی فشرده و کمبود دامنهٔ بیان |
| M7 | Gate انتخاب Expert مستقل از دامنهٔ هندسی، واریانس جدا، کالیبراسیون Validation | ازکارافتادن Bias Expert و undercoverage |

### تضمین دامنهٔ جدید

برای پایهٔ $c_{ui}\in[r_{min},r_{max}]$، جهت $s\in[-1,1]$ و Gain $g\in[0,1]$:

$$
\hat r_{ui}=c_{ui}+g_{ui}\,s_{ui}
\begin{cases}
r_{\max}-c_{ui},&s_{ui}\ge 0\\
c_{ui}-r_{\min},&s_{ui}<0
\end{cases}
$$

در نتیجه $r_{min}\le\hat r\le r_{max}$ **بدون Output Clipping**. این کران **کران حاصل‌ضرب شعاع‌های MBMF قدیم نیست**. متریک‌های مبتنی بر کران فردی قدیمی برای V2 مقدار NA دارند و Metric مستقل `v2_max_global_range_violation` گزارش می‌شود.

### اعتبار ارزیابی

- `train` فقط برای آموزش؛ Validation برای انتخاب Checkpoint و Configuration و کالیبراسیون واریانس Method7.
- برای Method7، زیرمجموعهٔ Fit کالیبراسیون و زیرمجموعهٔ گزارش NLL از Validation به صورت تعیین‌پذیر و غیرهمپوشان انتخاب می‌شوند.
- هرگونه مقایسهٔ ظرفیت، تنظیم LR و وزن Loss فقط روی Validation مجاز است؛ Test باید قفل بماند.
- برای مقاله به حداقل چند Split و چند Seed مستقل، Ablation و منحنی کالیبراسیون نیاز داریم.
- تغییر معماری هفت مدل باعث شکستن هم‌ارزی عددی Checkpointهای V1 شده و نباید مدل V1 را با V2 بارگذاری کرد.

### میزان اعتماد به تست‌ها

`22 passed` روی آزمون‌های **Unit/Mock محلی**: کران ریاضی، Gradients، قطع گرادیان شاخهٔ حذف‌شده، Fusion Gate، کالیبراسیون، Checkpoint و آزمون‌های V9. هفت تست یکپارچهٔ واقعی Synthetic روی کلاس‌های V7/V8 هم در GitHub Actions پاس شده‌اند؛ اجرای کامل MovieLens هنوز انجام نشده، پس **نتیجهٔ عملکرد تجربی V2 فعلاً نامعلوم است**.

### محدودیت‌ها و برنامهٔ لازم پیش از انتشار Q1

1. آزمون Gradient Conflict بین Reconstruction/Prediction در همهٔ هفت مدل.
2. K/Capacity/LR/Weight Decay با بودجه و Split/Seed یکسان برای روش‌ها و Baselineها.
3. پرهیز از گزارش کران شعاعی سابق برای روش‌های دارای Bias Residual جدید.
4. بازتولید چند-Seed و تحلیل Calibration (PICP/MPIW/NLL) Method7.
5. گزارش زمان اجرا و حافظه؛ استفاده از پروفایل `capacity` صرفاً در صورت بهبود پایدار Validation.

**Commit/Merge روی شاخهٔ `v2` نیازمند فرمان صریح کاربر در پیام بعدی است.**
