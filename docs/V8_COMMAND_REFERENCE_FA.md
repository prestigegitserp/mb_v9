# مرجع فرمان‌های نسخه۸ و تطبیق با راهنمای نسخه۷

این مرجع مکمل [راهنمای انتشار](V8_RELEASE_RUNBOOK_FA.md) و [راهنمای جامع](V8_RUNBOOK_FA.md) است. معادل‌ها از روی CLI واقعی بررسی شده‌اند. فایل‌های راهنمای نسخه۷ همچنان بدون تغییر موجودند.

## تطبیق ورودی‌ها و فایل‌ها

| نسخه۷ | نسخه۸ | توضیح |
|---|---|---|
| شاخهٔ v7 | شاخهٔ v8 | نسخهٔ انتشار برای اجرا |
| v7-development | v8_development | شاخهٔ توسعه؛ نام دقیق با underscore |
| paper_v7.py | paper_v8.py | راه‌انداز مرحله‌ای؛ انتخاب پیشنهادی برای اجرا |
| run_v7.py | run_v8.py | CLI مستقیم با گزینه‌های جزئی‌تر |
| requirements_v7.txt | requirements_v8.txt | محیط وابستگی نسخه۸ |
| verify_v7.py | verify_v8.py | علاوه بر نسخه۸، تمام۵۴۸ فایل درخت نسخه۷ را بررسی می‌کند |
| examples/v7/{profile}/{dataset}.json | examples/v8/{profile}/{dataset}.json | قالب ورودی config |
| estimate_v7.py | estimate_v8.py | شمارش برنامه و تخمین غیرقطعی |
| replot_v7.py | replot_v8.py | رسم دوباره از جدول بدون آموزش |
| --v7-device / --v7-config / --v7-* | --v8-device / --v8-config / --v8-* | در run_v8؛ راه‌انداز paper از نام کوتاه استفاده می‌کند |
| V7Light / V7Paper | V8Light / V8Paper | آرایه‌های PowerShell در راهنمای انتشار |
| final | test یا final | هر دو مرحلهٔ TEST برندگان scan هستند |

پیش‌فرض راه‌انداز paper_v7، پروفایل development بود؛ پیش‌فرض paper_v8، پروفایل common است. مثال‌های این راهنما پروفایل را صریح انتخاب می‌کنند. برای بازسازی پروتکل development قبلی، در ابتدای مطالعه `--profile development` بدهید؛ برای مقایسهٔ common با paper، یکسان‌بودن TEST را فرض نکنید.

## هر stage چه می‌کند؟

در ستون فرمان، `...` جای آرگومان نیست؛ شکل اجرایی کامل برای مراحل بعد از plan ذخیره‌شده، در بلوک پایین جدول آمده است.

| stage در paper_v8 | فرمان زیرین run_v8 | آموزش دارد؟ | داده و پیش‌نیاز |
|---|---|---|---|
| plan | plan | خیر | گرید و تعداد کارها؛ recipe را ذخیره می‌کند |
| smoke | main با quick | بله، کوتاه | TRAIN/VAL و TEST کوتاه؛ خروجی جدا |
| pilot | main | بله | config ثابت؛ TEST دارد؛ search ندارد |
| develop | scan بدون scan-only | بله | scan روی VAL، سپس final روی TEST |
| scan | scan با scan-only | بله | فقط TRAIN/VAL؛ بدون TEST |
| test / final | final | بله یا reuse | search و منشأ winner لازم است |
| lossstudy | lossstudy | بله | فقط VAL؛ پیرامون winner؛ scan لازم است |
| ablation | ablation | بله | حذف مؤلفه از winner؛ TEST توصیفی |
| ablation-fixed | ablation-fixed | بله | مقایسهٔ حذف مؤلفه با K ثابت |
| kstudy | kstudy | بله | Kهای تعریف‌شدهٔ مطالعه؛ TEST برای انتخاب مجدد نیست |
| sparsity | sparsity | بله | نگهداری بخشی از TRAIN؛ heldout ثابت |
| density | density | بله | فقط synthetic؛ جمعیت heldout می‌تواند تغییر کند |
| convergence | convergence | بله | TRAIN/VAL؛ schedule مرجع؛ TEST ندارد |
| calibration | calibration | بله، اگر فعال | متد۷ common؛ انتخاب با VAL NLL، TEST جدا |
| temporal | temporal | بله | فقط MovieLens؛ scan و TEST زمانی مستقل |
| report | report | خیر | گزارش از آرشیو موجود؛ با plots رسم را فعال کنید |
| curves | curves | خیر | منحنی history موجود؛ RMSE ثبت‌نشده تولید نمی‌کند |
| audit | audit | خیر | کنترل منشأ winner و پوشش TEST مطالعهٔ tuned |
| status | خواندن checklist | خیر | وضعیت تکمیل؛ حتی پیش از آموزش قابل اجراست |
| all | چند فرمان مستقل | بله | scan، TEST، lossstudy، دو ablation، K، sparsity، convergence، calibration، report، audit؛ density برای synthetic و temporal در صورت درخواست |

بعد از تعریف مطالعه و اجرای plan، برای نمونه:

```text
python paper_v8.py --stage scan --output ./results_v8/tuned_light
python paper_v8.py --stage lossstudy --output ./results_v8/tuned_light --loss-methods Method6 --loss-parameters transport_moment_weight
python paper_v8.py --stage test --output ./results_v8/tuned_light
python paper_v8.py --stage ablation --output ./results_v8/tuned_light
python paper_v8.py --stage ablation-fixed --output ./results_v8/tuned_light
python paper_v8.py --stage kstudy --output ./results_v8/tuned_light
python paper_v8.py --stage sparsity --output ./results_v8/tuned_light
python paper_v8.py --stage convergence --output ./results_v8/tuned_light
python paper_v8.py --stage calibration --output ./results_v8/tuned_light
python paper_v8.py --stage temporal --output ./results_v8/tuned_light
python paper_v8.py --stage report --output ./results_v8/tuned_light --plots
python paper_v8.py --stage curves --output ./results_v8/tuned_light --plots
python paper_v8.py --stage audit --output ./results_v8/tuned_light
python paper_v8.py --stage status --output ./results_v8/tuned_light
```

در مطالعهٔ تک‌K، kstudy منحنی چندنقطه‌ای نمی‌سازد. برای چند K از ابتدا recipe تازه با `--k 10,20,50` تعریف کنید. همهٔ خطوط بالا را برای هر پرسش پژوهشی لازم ندانید؛ دامنهٔ مقاله تعیین می‌کند کدام کنترل ضروری است. فهرست status هم‌زمان همهٔ شواهد احتمالی را نشان می‌دهد، نه حکم قبول/رد مقاله.

## معنی گزینه‌های مهم و محل درست استفاده

| گزینه | کجا؟ | معنی |
|---|---|---|
| --profile common/development/paper | paper_v8 | انتخاب پروتکل و قالب بودجه؛ study تازه |
| --budget light | paper_v8 | سقف۱۲۰epoch،۲step،۲۰۰epoch baseline و۶job، مگر مقدار صریح داده شود |
| --budget standard | paper_v8 | مقدارهای قالب هر روش حفظ می‌شود |
| --budget full | paper_v8 | سقف تعداد job گرید تعریف‌شده برداشته می‌شود؛ بهینهٔ جهانی نیست |
| --search-jobs-per-method N | paper_v8 | سقف configهای هر خانواده؛ صفر بدون سقف |
| --v8-search-jobs-per-method N | run_v8 | معادل مستقیم گزینهٔ بالا |
| --max-epochs / --steps-per-epoch / --baseline-max-epochs | paper_v8 | سقف‌های صریح؛ در run با پیشوند v8 |
| --split-seeds | هر دو | تکرار تقسیم داده؛ TRAIN/VAL/TEST را مشخص می‌کند |
| --model-seeds | هر دو | تکرار آموزش روی همان split |
| --manual-seed | هر دو | تعیین یک model seed و search seed؛ با model-seeds همزمان نیست |
| --k | هر دو | Kهای کاندید؛ ظرفیت autoencoderها گرید مستقل دارد |
| --deterministic | paper_v8 | کنترل تکمیلی از آغاز مطالعه؛ run: v8-deterministic |
| --device / --threads / --storage | paper_v8 | محل اجرا و ذخیره؛ run: با پیشوند v8 |
| --host-budget-mb / --ssd-budget-gb / --cache-dir | paper_v8 | سهم offload و مسیر cache؛ سقف کل RAM/خروجی نیست |
| --methods | هر دو | محدودکردن آگاهانهٔ مطالعه؛ اجرای معمول همهٔ روش‌های مجاز را دارد |
| --no-plots / --plots | paper_v8 | کنارگذاشتن رسم هنگام آموزش / فعال‌کردن رسم در report |
| --epoch-train-rmse | هر دو | RMSE کامل TRAIN در تیک‌های ارزیابی؛ هزینهٔ اضافه دارد |
| --loss-methods / --loss-parameters | هر دو | محدودکردن lossstudy؛ برندهٔ اصلی را عوض نمی‌کند |
| --include-temporal | paper_v8 | اضافه‌کردن temporal به all برای MovieLens |
| --resume | هر دو | در paper برای مراحل آموزش خودکار است |
| --dry-run | هر دو | ساخت/نمایش برنامه بدون آموزش؛ در paper recipe ذخیره نمی‌شود |
| --scan-only | فقط run_v8 scan | توقف پیش از TEST؛ paper stage scan آن را خودکار می‌دهد |

روی یک recipe موجود، تنظیم علمی جدید را تحمیل نکنید. گزینه‌های محل اجرا و رسم را می‌توان عوض کرد؛ برای بودجهٔ علمی، profile، K یا seed جدید، output تازه تعریف کنید. این جداسازی مانع اشتباه‌گرفتن winnerهای دو مطالعه می‌شود.

فرمان‌هایی که خود برنامه روی Windows چاپ می‌کند، با quoting لفظی PowerShell و call operator یعنی `&` هستند تا فاصله، apostrophe و `$` در مسیرها خراب نشوند. در Linux از quoting همان shell استفاده می‌شود. این نمایش با فهرست آرگومان واقعیِ اجرای subprocess متفاوت است؛ خود برنامه دستورها را به‌صورت آرگومان‌های جدا اجرا می‌کند.

## اجرای پیشرفته با CLI مستقیم

برای تنظیم نرخ‌های sparsity/density یا K ثابت ablation، CLI مستقیم این گزینه‌ها را نیز دارد: `--keep-rates`، `--density-rates` و `--ablation-fixed-k`. پارامترهای علمی دیگر، config و بودجه را دقیقاً مطابق مطالعهٔ مرجع تکرار کنید. برای استفادهٔ معمول، فرمان paper و recipe ذخیره‌شده ساده‌تر است.

برای خودآزمایی داخلی و مشاهدهٔ کمک:

```text
python run_v8.py self-test --datasets ml-100k
python paper_v8.py --help
python run_v8.py --help
python estimate_v8.py --help
python replot_v8.py --help
```

کنترل اختیاری blend دو روش، همان قابلیت قبلی است و روش نهم پیشنهادی ایجاد نمی‌کند. برای تعریف پیشاپیش AutoRec+Method4 به‌عنوان کنترل ensemble:

```text
python paper_v8.py --stage plan --profile common --budget light --datasets ml-100k --data-root ./datasets --output ./results_v8/blend_control --device cuda --blend-methods AutoRec,Method4
python paper_v8.py --stage develop --output ./results_v8/blend_control
```

## خطاهای رایج و پاسخ عملی

| مشاهده | معنی / اقدام |
|---|---|
| scan تمام شد ولی TEST نیست | paper scan عمداً VAL-only است؛ stage test را روی همان output اجرا کنید |
| فقط شکل search دارم | مربوط به VAL است؛ جدول final و publication ریشه را بررسی کنید |
| No validation winners | ابتدا scan همان مطالعه باید کامل شود |
| Existing study / DIFFERENT config | تنظیمات علمی یا سورس با مطالعهٔ قبلی فرق دارد؛ output تازه |
| CUDA unavailable | کرنل Python فعلی CUDA را نمی‌بیند؛ محیط را بررسی یا device auto انتخاب کنید |
| یک split و warning آماری | جدول توصیفی معتبر است؛ آزمون معناداری بین splitها ساخته نمی‌شود |
| calibration در paper skipped | متد۷ point-only است؛ برای احتمال مطالعهٔ common جدا تعریف کنید |
| audit روی pilot رد می‌شود | audit انتخاب tuned به scan و TEST منتخب نیاز دارد؛ برای پایلوت status را ببینید |
| K study یک نقطه دارد | در recipe فقط یک K تعریف شده است |
| منحنی TRAIN خالی است | RMSE کامل TRAIN ثبت نشده؛ convergence یا انتخاب epoch-train-rmse از ابتدا |
| report نمودار TEST نمی‌سازد | آرشیو TEST موجود نیست؛ report آموزش یا TEST را جایگزین نمی‌کند |
| پس از توقف دوباره شروع کردم | همان stage/output را تکرار کنید؛ بازیابی فقط برای حالت معتبر همان مطالعه است |

شواهد عددی، محدودیت GPU و نتیجهٔ منفی متد۳ در [گزارش اعتبارسنجی](V8_VALIDATION_FA.md) حفظ شده‌اند. این اصلاح انتشار، ادعای تازه‌ای دربارهٔ بهبود TEST تولید نمی‌کند.
