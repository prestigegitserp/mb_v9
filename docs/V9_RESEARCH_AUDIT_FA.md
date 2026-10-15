# ممیزی روش‌شناختی V9 — قرارداد آزمایش‌های Q1

## Provenance

- مبنا: `prestigegitserp/mb_v8`, شاخهٔ `fix/windows-crlf-core-hash`, commit `f58ec7b95ddc802561cbc1399bcc17be226b10de`.
- هستهٔ علمی: `mbmf_q1_final_v7.py`, نرمال‌سازی پایان خطوط CRLF به LF، SHA-256:
  `6da726cb155ab6ad291f5f608febc37f42d1c25ac3924910f56917908c1f124c`.
- زیرساخت V8، Git blob: `6319902dc48385661c959472fd5edee3daf51f51`.
- انتقال هستهٔ علمی/زیرساخت با `bootstrap_v8.py` به `vendor_v8/`. ریپوی V8 فقط خوانده می‌شود. هر تغییر فرمول باید در پژوهش جداگانه با SHA جدید و رجیستری پروتکل جدید باشد.

## اشکالات و نقاط خطر یافته‌شده

1. **اسکنِ یک‌پارامتری محدود:** اسکن اصلی فقط One-Factor است؛ اثر متقابل gamma/consistency، alpha/beta، context/tau و ... را کامل نمی‌بیند. V9 زوج‌های منتخب را اضافه می‌کند. **فضای کامل دکارتی همهٔ پارامترها نیست.**
2. **انتقال تنظیمات:** `--hyper-scan` در V8 مستقل از اجرای اصلی است و خودکار تنظیم برندگان را اعمال نمی‌کند. V9 پس از validation-only scan، ترکیب برندهٔ **هر روش و هر Outer Split** را با K و Initialization یکسان در اجرای Test استفاده می‌کند.
3. **نشت غیرمستقیم میان Splitها:** میانگین Validation همهٔ split-seedها برای انتخاب **یک پیکربندی جهانی** مشکل‌زا است: تعامل Test در یک Split ممکن است در Train/Val Split دیگر باشد. بنابراین V9 **به هیچ وجه از میانگین برون-Split برای انتخاب برنده استفاده نمی‌کند**؛ تصمیم فقط از Validation همان Split و Model Seedهای آن گرفته می‌شود.
4. **تعداد اجراهای برابر:** Winner فقط اگر همهٔ Model Seedهای مقرر را داشته باشد، معتبر است. کاندید ناقص کنار گذاشته نمی‌شود تا برندهٔ غلط اعلام شود؛ انتخاب با خطا متوقف می‌شود.
5. **Test Leakage مستقیم:** در مرحلهٔ tune، وجود حتی یک ستون `test_` ممنوع است. Test فقط بعد از قفل‌شدن config همان Outer Split ارزیابی می‌شود.
6. **کاندیدهای Data Protocol:** `train_fraction=0.90` در V8 با `val_fraction=0.10` موجب `test_fraction=0` می‌شود. علاوه بر این تغییر train_fraction در کنار test_fraction مقایسه را مخدوش می‌کند. V9 این jobها را از تنظیم ابرپارامتر حذف و تحلیل مستقل `train-only sparsity` با Test/Val ثابت جایگزین می‌کند.
7. **تراکم Synthetic:** ورودی Synthetic با نرخ مشاهدهٔ متفاوت، مجموعه‌های مشاهدات و Test متفاوتی تولید می‌کند؛ بنابراین Density Study به‌وضوح *توصیفی/حساسیت* برچسب‌گذاری شده، نه جست‌وجوی ابرپارامتر.
8. **Controlled Ablation:** انتخاب آزاد K برای هر Ablation می‌تواند اثر مکانیزم را با تغییر ظرفیت مخلوط کند؛ V9 یک مطالعهٔ مستقل `ablation-fixed` با K یکسان برای تمام نسخه‌های Full/Ablated افزوده است.
9. **K study:** در V8 فقط کاندیدهای منتخب روی Test اندازه‌گیری می‌شوند. برای نمودار Train/Val/Test RMSE با K ثابت باید هر K مطالعهٔ مستقل داشته باشد. V9 انجام می‌دهد اما **Test برای انتخاب K مجاز نیست**؛ MBMF Initialization در هر K همچنان با Validation همان Split انتخاب می‌شود.
10. **Convergence:** V8 بیشتر Loss و `val_RMSE` را در هر `eval_every` ثبت می‌کند، نه RMSE سه Split در هر Epoch. V9 با ابزار اندازه‌گیری `train_RMSE` را در همان تیک‌ها قابل‌فعال‌سازی می‌کند. هیچ Test metric در حلقهٔ epoch برای انتخاب مدل محاسبه نمی‌شود. MBMF Sparse همچنان تابع هدف را به جای Validation RMSE در تاریخچهٔ Epoch می‌دهد؛ این محدودیت پنهان نشده است.
11. **Method7 Calibration:** انتخاب صرفاً با RMSE می‌تواند کالیبراسیون NLL را تضعیف کند؛ V9 یک انتخاب مستقل `calibration` برای Method7 دارد که با `val_Gaussian_NLL` همان Split انجام و جدا از نتیجهٔ RMSE-primary گزارش می‌شود.
12. **K مخصوص دو Autoencoder:** در V8، AutoRec و AEMC-NE ابعاد نهفتهٔ مستقلی دارند و در K loop عمومی فقط یک بار اجرا می‌شوند؛ نباید آنها را در نمودار K همانند MF در نظر گرفت. ظرفیت AutoRec/AEMC جداگانه روی Validation اسکن می‌شود.
12. **نابرابری بودجهٔ tuning:** Methods 1–7 و رقبا ابرپارامترهای متفاوت دارند و شمار کاندیدهای یکسان ندارند؛ V9 تعداد job و تعداد candidate هر روش را در `V9_SEARCH_PLAN.json` ثبت می‌کند. برای نتیجهٔ قوی Q1، نتایج باید همراه بودجهٔ آزمایش هر روش و زمان tuning گزارش شوند. برابر بودن تعداد کاندیدها به‌تنهایی منصفانه‌بودن را تضمین نمی‌کند.
13. **Cross-dataset RMSE:** مقیاس Jester [-10,10] با ML [1,5] فرق دارد، لذا RMSE خام میان دیتاست‌ها نباید میانگین غیرنرمال‌شده شود. خلاصه‌های اصلی هر دیتاست جدا گزارش شده‌اند.
14. **Ranking:** معیارهای NDCG/MAP/MRR در V8 روی تعامل‌های امتیازخوردهٔ heldout محاسبه می‌شوند، نه ranking کل کاتالوگ. گزارش مقاله باید دقیقاً این مسئله را ذکر کند.
15. **External baselines:** BMC-ADMM، Deep-MMMF، DRPRA به author-code و splits مشترک نیاز دارند؛ V9 خروجی آنها را تخمین یا به‌جای کد رسمی اجرا نمی‌کند.
16. **Resume:** فقط job `COMPLETED` با fingerprint مطابق و فایل خروجی موجود قابل skip است. job نیمه‌تمام از ابتدا آن job اجرا می‌شود؛ Resume تا وسط epoch یا مدل پیاده نشده است. فایل نیمه‌تمام را به‌جای نتیجهٔ کامل گزارش نکنید.

## نتیجهٔ مورد انتظار و درجهٔ اطمینان

V9 به‌لحاظ طراحی انتخاب validation-only را enforce می‌کند؛ ولی تا زمانی که یک اجرای Integration روی clone ثابت V8 و دیتاست‌های واقعی انجام نشده، **ادعای عملکرد عددی، سرعت یا رفع قطعی همهٔ خطاها مجاز نیست**. آزمون‌های واحد موجود فقط پوشش قوانین انتخاب و invariance مربوط به train subsampling را فراهم می‌کنند.

## فازهای پیشنهادی اجرای مقاله

1. `python run_v9.py self-test` و `python -m pytest -q tests`؛
2. Pilot محدود `python run_v9.py main --datasets ml-100k --quick`، جدا از نتایج مقاله؛
3. `python run_v9.py plan` برای بودجه؛
4. `scan` با تمام Seedها؛
5. `final` با تنظیمات Validation-frozen در هر Split؛
6. `ablation` و `kstudy` (فقط تفسیر / تحلیل پایداری)؛
7. `sparsity` و `density`؛
8. `convergence` (Train/Validation در epoch، نه Test)؛
9. `temporal` برای MovieLens اگر لازم شد؛
10. `audit`، نمودارها، Wilcoxon-Holm، Friedman/Nemenyi و بررسی عددی دستی.
