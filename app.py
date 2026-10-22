import streamlit as st
import matplotlib.pyplot as plt
from sentinel.scanner import Scanner

# ==========================================
# تنظیمات صفحه
# ==========================================
st.set_page_config(
    page_title="URLSentinel | تحلیل امنیتی URL",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ URLSentinel")
st.markdown("""
**تشخیص فیشینگ با شبکه عصبی از صفر (NumPy) و هوش مصنوعی توضیح‌پذیر**

این دموی زنده، یک URL را بدون باز کردن آن تحلیل می‌کند و احتمال فیشینگ بودن آن را همراه با **دلایل تصمیم‌گیری** نشان می‌دهد.
""")

# ==========================================
# بارگذاری مدل (فقط یک بار انجام می‌شود)
# ==========================================
@st.cache_resource
def load_scanner():
    return Scanner()

try:
    scanner = load_scanner()
    st.success("✅ مدل با موفقیت بارگذاری شد.")
except Exception as e:
    st.error(f"خطا در بارگذاری مدل: {e}")
    st.info("مطمئن شو که فایل `models/sentinel_mlp.npz` وجود دارد.")
    st.stop()

# ==========================================
# رابط کاربری
# ==========================================
url_input = st.text_input("URL مورد نظر را وارد کنید:", placeholder="https://example.com/login")

if st.button("تحلیل امنیتی", type="primary"):
    if not url_input:
        st.warning("لطفاً یک URL وارد کنید.")
    else:
        with st.spinner("در حال تحلیل ساختار URL..."):
            try:
                result = scanner.scan(url_input)
                prob = result["phishing_probability"]
                verdict_text = result["verdict"]
                reasons = result["reasons"]

                col1, col2 = st.columns([1, 1])

                # ستون اول: نتیجه نهایی
                with col1:
                    st.metric("امتیاز ریسک فیشینگ", f"{prob:.1%}")
                    if verdict_text == "PHISHING":
                        st.error("🚨 **PHISHING**: این URL به احتمال زیاد فیشینگ است.")
                    elif verdict_text == "SUSPICIOUS":
                        st.warning("⚠️ **SUSPICIOUS**: این URL مشکوک است، با احتیاط رفتار کنید.")
                    else:
                        st.success("✅ **SAFE**: این URL امن به نظر می‌رسد.")

                # ستون دوم: دلایل (توضیح‌پذیری)
                with col2:
                    st.subheader("چرا این تصمیم گرفته شد؟")
                    st.caption("دلایل برگرفته از مقادیر SHAP / impact")

                    risk_reasons = [r for r in reasons if r["direction"] == "risk"]
                    safe_reasons = [r for r in reasons if r["direction"] == "safe"]

                    if risk_reasons:
                        st.markdown("**🔴 عوامل افزایش‌دهنده ریسک:**")
                        for r in risk_reasons:
                            st.write(f"- {r['reason']} (تأثیر: `{r['impact']}`)")

                    if safe_reasons:
                        st.markdown("**🟢 عوامل کاهش‌دهنده ریسک:**")
                        for r in safe_reasons:
                            st.write(f"- {r['reason']} (تأثیر: `{r['impact']}`)")

                    # نمودار میله‌ای برای دلایل
                    if reasons and reasons[0]["feature"] is not None:
                        fig, ax = plt.subplots(figsize=(6, 3))
                        labels = [r["reason"] for r in reasons]
                        values = [r["impact"] for r in reasons]
                        colors = ['red' if r["direction"] == "risk" else 'green' for r in reasons]

                        ax.barh(labels, values, color=colors)
                        ax.axvline(0, color='black', linewidth=0.8)
                        ax.set_xlabel("تأثیر بر تصمیم")
                        ax.invert_yaxis()
                        st.pyplot(fig)

                with st.expander("مشاهده ویژگی‌های استخراج‌شده از URL"):
                    st.json(result["features"])

            except Exception as e:
                st.error(f"خطا در تحلیل URL: {e}")

st.divider()
st.caption("ساخته شده با ❤️ با استفاده از Streamlit | مدل: شبکه عصبی پیاده‌سازی شده از صفر با NumPy")