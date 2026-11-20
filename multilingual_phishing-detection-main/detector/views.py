import os
import re
import io
import pickle
from datetime import timedelta
from functools import wraps
from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse, Http404
from django.utils import timezone
from django.utils.html import escape
from django.utils.safestring import mark_safe
from django.db.models import Avg
from django.views.decorators.http import require_POST
from django.contrib.auth import login, logout
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.models import User
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from .models import ScanHistory

MODEL_PATH = os.path.join(os.path.dirname(__file__), "lstm_model.keras")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "tokenizer.pkl")

model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 100

TRUSTED_DOMAINS = [
    "capgemini.com", "jobs2web.com", "google.com", "microsoft.com",
    "amazon.com", "linkedin.com", "github.com", "apple.com",
    "naukri.com", "indeed.com", "sbi.co.in", "hdfcbank.com",
    "icicibank.com", "axisbank.com", "jobsora.com", "glassdoor.com",
    "monster.com", "shine.com", "timesjobs.com", "foundit.in",
    "instahyre.com", "internshala.com",
]

URGENCY_WORDS = [
    "urgent", "immediately", "verify your account", "suspended",
    "click here", "act now", "limited time", "confirm your identity",
    "your account will be closed", "password expired", "winner",
    "claim your prize", "security alert", "verify your identity",
    "failure to act", "account will be suspended", "unusual activity",
    "உடனடியாக", "கணக்கு முடக்கப்படும்", "சரிபார்க்கவும்", "எச்சரிக்கை",
    "వెంటనే", "ఖాతా నిలిపివేయబడుతుంది", "ధృవీకరించండి", "హెచ్చరిక",
    "ഉടൻ", "അക്കൗണ്ട് സസ്പെൻഡ്", "സ്ഥിരീകരിക്കുക", "മുന്നറിയിപ്പ്",
]

FINANCIAL_KEYWORDS = [
    "bank account", "credit card", "debit card", "otp", "cvv", "pin number",
    "net banking", "upi", "ifsc", "kyc", "refund", "transaction failed",
    "payment pending", "wire transfer", "crypto", "bitcoin", "loan approved",
    "வங்கி கணக்கு", "బ్యాంక్ ఖాతా", "ബാങ്ക് അക്കൗണ്ട്",
]

SUSPICIOUS_KEYWORDS_IN_DOMAIN = ["verify", "secure", "login", "update", "account", "confirm"]


# ======================= access control =======================

def user_required(view_func):
    """Only logged-in NORMAL users. Admin is sent to the admin dashboard."""
    @wraps(view_func)
    def _wrapped(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect('user_login')
        if request.user.is_staff:
            return redirect('admin_dashboard')
        return view_func(request, *args, **kwargs)
    return _wrapped


def admin_required(view_func):
    """Only logged-in STAFF (admin) accounts."""
    @wraps(view_func)
    def _wrapped(request, *args, **kwargs):
        if not request.user.is_authenticated or not request.user.is_staff:
            return redirect('admin_login')
        return view_func(request, *args, **kwargs)
    return _wrapped


def any_login_required(view_func):
    """Any logged-in account (user or admin)."""
    @wraps(view_func)
    def _wrapped(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect('home')
        return view_func(request, *args, **kwargs)
    return _wrapped


def get_scan_for_user(request, scan_id):
    """Normal users can open only their own scans. Admin can open any."""
    scan = get_object_or_404(ScanHistory, id=scan_id)
    if not request.user.is_staff and scan.user_id != request.user.id:
        raise Http404("Scan not found")
    return scan


# ======================= detection helpers =======================

def detect_language(text):
    tamil = len(re.findall(r'[\u0B80-\u0BFF]', text))
    malayalam = len(re.findall(r'[\u0D00-\u0D7F]', text))
    telugu = len(re.findall(r'[\u0C00-\u0C7F]', text))
    counts = {"Tamil": tamil, "Malayalam": malayalam, "Telugu": telugu}
    best_lang = max(counts, key=counts.get)
    if counts[best_lang] == 0:
        return "English"
    return best_lang


def extract_sender_domain(text):
    match = re.search(r'[\w\.\-]+@([\w\-]+\.[\w\.\-]+)', text)
    return match.group(1).lower() if match else None

def is_trusted_domain(domain):
    if not domain:
        return False
    return any(domain == d or domain.endswith("." + d) for d in TRUSTED_DOMAINS)

def sender_risk_flags(domain):
    flags = []
    if not domain:
        flags.append("No sender email detected")
        return flags
    if is_trusted_domain(domain):
        flags.append(f"Sender domain '{domain}' is a verified trusted domain")
    else:
        flags.append(f"Sender domain '{domain}' is NOT in trusted list")
        if any(k in domain for k in SUSPICIOUS_KEYWORDS_IN_DOMAIN):
            flags.append("Sender domain contains suspicious keyword")
        if domain.count("-") >= 2:
            flags.append("Sender domain has unusual hyphen pattern")
    return flags

def unknown_sender_pattern(domain, trusted_flag):
    if not domain:
        return True
    if trusted_flag:
        return False
    if re.search(r'\d{3,}', domain) or len(domain.split('.')[0]) > 20:
        return True
    return False


def extract_urls(text):
    return re.findall(r'https?://([\w\-\.]+)', text)

def url_threat_flags(urls):
    flags = []
    if not urls:
        flags.append("No URLs found in email")
        return flags, False
    suspicious = False
    for u in urls:
        if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', u):
            flags.append(f"URL uses raw IP address ({u}) — high risk")
            suspicious = True
        elif not is_trusted_domain(u):
            flags.append(f"URL domain '{u}' is not in trusted list")
            suspicious = True
        else:
            flags.append(f"URL domain '{u}' is trusted")
    return flags, suspicious

def url_analysis_pipeline(urls, trusted_domains_list):
    steps = []
    if not urls:
        steps.append({"stage": "URL Detected", "result": "No URL found", "risk": False})
        return steps

    steps.append({"stage": "URL Detected", "result": f"{len(urls)} URL(s) found: {', '.join(urls[:2])}", "risk": False})

    domain = urls[0]
    is_trusted = any(domain == d or domain.endswith("." + d) for d in trusted_domains_list)
    steps.append({
        "stage": "Domain Analysis",
        "result": f"'{domain}' is {'a trusted' if is_trusted else 'an untrusted'} domain",
        "risk": not is_trusted
    })

    suspicious_keywords = ["verify", "secure", "login", "update", "account", "confirm"]
    is_ip = bool(re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', domain))
    has_keyword = any(k in domain for k in suspicious_keywords)
    many_hyphens = domain.count("-") >= 2
    pattern_found = is_ip or has_keyword or many_hyphens
    reasons = []
    if is_ip: reasons.append("raw IP address")
    if has_keyword: reasons.append("suspicious keyword")
    if many_hyphens: reasons.append("excessive hyphens")
    steps.append({
        "stage": "Suspicious Pattern",
        "result": ", ".join(reasons) if reasons else "No suspicious pattern detected",
        "risk": pattern_found
    })

    if is_trusted and not pattern_found:
        classification, risk4 = "Safe URL", False
    elif pattern_found:
        classification, risk4 = "High-Risk URL", True
    else:
        classification, risk4 = "Unverified URL", True
    steps.append({"stage": "Threat Classification", "result": classification, "risk": risk4})

    url_risk_score = 0
    if not is_trusted: url_risk_score += 30
    if pattern_found: url_risk_score += 50
    if is_ip: url_risk_score += 20
    url_risk_score = min(url_risk_score, 100)
    steps.append({"stage": "Risk Score", "result": f"{url_risk_score}/100", "risk": url_risk_score >= 50})

    return steps


def urgency_flags(text):
    return [w for w in URGENCY_WORDS if w in text.lower() or w in text]

def financial_flags(text):
    return [w for w in FINANCIAL_KEYWORDS if w in text.lower() or w in text]

def highlight_text(text, words):
    escaped = escape(text)
    for w in sorted(set(words), key=len, reverse=True):
        if not w:
            continue
        pattern = re.escape(escape(w))
        escaped = re.sub(f'({pattern})', r'<mark style="background:#fef08a; padding:1px 3px; border-radius:3px;">\1</mark>', escaped, flags=re.IGNORECASE)
    return mark_safe(escaped)

def get_safety_recommendation(verdict):
    if verdict == "High Risk":
        return {"title": "⚠ HIGH RISK MESSAGE", "points": [
            "Do not click the attached link",
            "Do not share OTP/password",
            "Verify the sender independently",
            "Report the message as spam",
        ]}
    elif verdict == "Medium Risk":
        return {"title": "⚠ MEDIUM RISK MESSAGE", "points": [
            "Avoid clicking links until verified",
            "Do not share personal or financial details",
            "Contact the sender through an official channel",
            "Mark as suspicious if unsure",
        ]}
    elif verdict == "Low Risk":
        return {"title": "⚠ LOW RISK MESSAGE", "points": [
            "Double-check the sender's email address",
            "Avoid sharing sensitive information",
            "Proceed with normal caution",
        ]}
    else:
        return {"title": "✅ SAFE MESSAGE", "points": [
            "No suspicious indicators found",
            "Sender domain is verified and trusted",
            "No immediate action needed",
        ]}


def build_analytics(qs, scope):
    total = qs.count()
    phishing_detected = qs.filter(result="Phishing").count()
    legitimate_count = qs.filter(result="Legitimate").count()
    high_risk_count = qs.filter(verdict="High Risk").count()
    avg_risk = round(qs.aggregate(avg=Avg('threat_score'))['avg'] or 0, 1)

    language_accuracy = [
        {"language": "Tamil", "accuracy": 100.0, "samples": 241},
        {"language": "Malayalam", "accuracy": 100.0, "samples": 257},
        {"language": "Telugu", "accuracy": 100.0, "samples": 252},
    ]
    return {
        "scope": scope,
        "total": total,
        "phishing_detected": phishing_detected,
        "legitimate_count": legitimate_count,
        "high_risk_count": high_risk_count,
        "avg_risk": avg_risk,
        "language_accuracy": language_accuracy,
    }


# ======================= landing + authentication =======================

def home(request):
    """Landing page: choose User or Admin."""
    if request.user.is_authenticated:
        return redirect('admin_dashboard' if request.user.is_staff else 'predict_email')
    return render(request, "detector/home.html")


def user_login(request):
    if request.user.is_authenticated and not request.user.is_staff:
        return redirect('predict_email')
    error = None
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            account = form.get_user()
            if account.is_staff:
                error = "This is an admin account. Please use the Admin Login."
            else:
                login(request, account)
                return redirect('predict_email')
        else:
            error = "Invalid username or password."
    return render(request, "detector/login.html", {
        "role": "user", "error": error,
        "username": request.POST.get("username", ""),
    })


def admin_login(request):
    if request.user.is_authenticated and request.user.is_staff:
        return redirect('admin_dashboard')
    error = None
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            account = form.get_user()
            if not account.is_staff:
                error = "This account does not have admin access. Please use the User Login."
            else:
                login(request, account)
                return redirect('admin_dashboard')
        else:
            error = "Invalid admin username or password."
    return render(request, "detector/login.html", {
        "role": "admin", "error": error,
        "username": request.POST.get("username", ""),
    })


def signup_view(request):
    if request.user.is_authenticated and not request.user.is_staff:
        return redirect('predict_email')
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            account = form.save()
            login(request, account)
            return redirect('predict_email')
    else:
        form = UserCreationForm()
    return render(request, "detector/signup.html", {"form": form})


def logout_view(request):
    logout(request)
    return redirect('home')


# ======================= USER area =======================

@user_required
def predict_email(request):
    context = {}

    if request.method == "POST":
        sender_input = request.POST.get("sender_email", "")
        subject_input = request.POST.get("subject", "")
        body_input = request.POST.get("body", "")
        email_text = f"{sender_input}\nSubject: {subject_input}\n{body_input}"

        language = detect_language(email_text)

        sender_domain = extract_sender_domain(email_text)
        trusted_flag = is_trusted_domain(sender_domain)
        sender_flags = sender_risk_flags(sender_domain)
        unknown_pattern = unknown_sender_pattern(sender_domain, trusted_flag)

        urls = extract_urls(email_text)
        url_flags, suspicious_url = url_threat_flags(urls)
        url_pipeline = url_analysis_pipeline(urls, TRUSTED_DOMAINS)

        seq = tokenizer.texts_to_sequences([email_text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
        ml_score = float(model.predict(padded, verbose=0)[0][0])

        urgency_found = urgency_flags(email_text)
        financial_found = financial_flags(email_text)

        strong_combo_risk = suspicious_url and len(urgency_found) >= 1
        heavy_urgency = len(urgency_found) >= 3

        if trusted_flag and not suspicious_url:
            base_result = "Legitimate"
            ml_confidence = max(round((1 - ml_score) * 100, 2), 85.0)
        elif strong_combo_risk or (not trusted_flag and heavy_urgency):
            base_result = "Phishing"
            combo_conf = 70 + (len(urgency_found) * 5) + (15 if suspicious_url else 0)
            ml_confidence = min(round(max(combo_conf, ml_score * 100), 2), 99.0)
        elif not trusted_flag and not suspicious_url and not urgency_found:
            base_result = "Legitimate" if ml_score < 0.85 else "Phishing"
            ml_confidence = round((ml_score if base_result == "Phishing" else 1 - ml_score) * 100, 2)
        else:
            base_result = "Phishing" if ml_score > 0.5 else "Legitimate"
            ml_confidence = round((ml_score if base_result == "Phishing" else 1 - ml_score) * 100, 2)

        reasons = []
        if urgency_found:
            reasons.append("Urgency detected")
        if suspicious_url:
            reasons.append("Suspicious URL")
        if financial_found:
            reasons.append("Financial keywords")
        if unknown_pattern:
            reasons.append("Unknown sender pattern")

        indicator_matrix = [
            {"check": "Sender domain trusted", "status": trusted_flag},
            {"check": "Suspicious/untrusted URL present", "status": suspicious_url},
            {"check": "Urgency/pressure wording found", "status": len(urgency_found) > 0},
            {"check": "Financial keywords found", "status": len(financial_found) > 0},
            {"check": "Unknown sender pattern", "status": unknown_pattern},
            {"check": "ML model flags as phishing", "status": ml_score > 0.5},
        ]

        explanation_parts = []
        if trusted_flag:
            explanation_parts.append(f"the sender domain ({sender_domain}) is verified and trusted")
        else:
            explanation_parts.append(f"the sender domain ({sender_domain or 'unknown'}) is not recognized as trusted")
        if suspicious_url:
            explanation_parts.append("the email contains links to unverified or suspicious domains")
        if urgency_found:
            explanation_parts.append(f"urgent/pressure language was detected ({', '.join(urgency_found[:3])})")
        if financial_found:
            explanation_parts.append(f"financial/sensitive keywords were detected ({', '.join(financial_found[:3])})")
        if ml_score > 0.5:
            explanation_parts.append(f"the deep learning model scored this content as {round(ml_score*100,1)}% phishing-like")
        else:
            explanation_parts.append(f"the deep learning model scored this content as {round((1-ml_score)*100,1)}% legitimate-like")
        ai_explanation = "This email was classified as " + base_result + " because " + "; ".join(explanation_parts) + "."

        score = ml_score * 30
        if not trusted_flag:
            score += 10
        if suspicious_url:
            score += 25
        if financial_found:
            score += 10
        score += min(len(urgency_found) * 8, 25)
        threat_score = min(round(score, 1), 100.0)

        if threat_score >= 70:
            verdict = "High Risk"
        elif threat_score >= 40:
            verdict = "Medium Risk"
        elif threat_score >= 15:
            verdict = "Low Risk"
        else:
            verdict = "Safe"

        recommendation = get_safety_recommendation(verdict)

        all_flags = sender_flags + url_flags + [f"Urgency word: {w}" for w in urgency_found] + [f"Financial word: {w}" for w in financial_found]
        scan = ScanHistory.objects.create(
            user=request.user,
            email_text=email_text[:2000],
            language=language,
            sender_domain=sender_domain or "unknown",
            ml_score=round(ml_score, 4),
            threat_score=threat_score,
            result=base_result,
            verdict=verdict,
            flags=", ".join(all_flags),
        )

        context = {
            "scan_id": scan.id,
            "sender_input": sender_input,
            "subject_input": subject_input,
            "body_input": body_input,
            "result": base_result,
            "confidence": ml_confidence,
            "language": language,
            "sender_domain": sender_domain,
            "trusted_flag": trusted_flag,
            "sender_flags": sender_flags,
            "url_flags": url_flags,
            "url_pipeline": url_pipeline,
            "urgency_found": urgency_found,
            "financial_found": financial_found,
            "reasons": reasons,
            "indicator_matrix": indicator_matrix,
            "ai_explanation": ai_explanation,
            "threat_score": threat_score,
            "verdict": verdict,
            "recommendation": recommendation,
        }

    return render(request, "detector/index.html", context)


@user_required
def scan_history(request):
    scans = ScanHistory.objects.filter(user=request.user).order_by('-created_at')[:50]
    return render(request, "detector/history.html", {"scans": scans})


@user_required
def threat_analytics(request):
    qs = ScanHistory.objects.filter(user=request.user)
    return render(request, "detector/analytics.html", build_analytics(qs, "Your scans"))


@user_required
@require_POST
def report_false_positive(request, scan_id):
    scan = get_scan_for_user(request, scan_id)
    scan.reported_false_positive = True
    scan.save(update_fields=["reported_false_positive"])
    return redirect('forensic_report', scan_id=scan.id)


# ======================= shared report (user: own, admin: any) =======================

@any_login_required
def forensic_report(request, scan_id):
    scan = get_scan_for_user(request, scan_id)

    urgency_found = urgency_flags(scan.email_text)
    financial_found = financial_flags(scan.email_text)
    urls = extract_urls(scan.email_text)
    url_flags, suspicious_url = url_threat_flags(urls)
    trusted_flag = is_trusted_domain(scan.sender_domain)
    unknown_pattern = unknown_sender_pattern(scan.sender_domain, trusted_flag)

    reasons = []
    if urgency_found:
        reasons.append("Urgency detected")
    if suspicious_url:
        reasons.append("Suspicious URL")
    if financial_found:
        reasons.append("Financial keywords")
    if unknown_pattern:
        reasons.append("Unknown sender pattern")

    threat_indicators = [
        {"check": "Sender domain trusted", "status": trusted_flag},
        {"check": "Suspicious/untrusted URL present", "status": suspicious_url},
        {"check": "Urgency/pressure wording found", "status": len(urgency_found) > 0},
        {"check": "Financial keywords found", "status": len(financial_found) > 0},
        {"check": "Unknown sender pattern", "status": unknown_pattern},
    ]

    highlighted = highlight_text(scan.email_text, urgency_found + financial_found)

    ai_explanation = (
        f"This email was classified as {scan.result} because the sender domain "
        f"({scan.sender_domain}) is {'verified and trusted' if trusted_flag else 'not recognized as trusted'}"
        f"{'; urgent/pressure language was detected' if urgency_found else ''}"
        f"{'; financial keywords were detected' if financial_found else ''}"
        f"{'; the email links to unverified domains' if suspicious_url else ''}."
    )

    context = {
        "scan": scan,
        "reasons": reasons,
        "threat_indicators": threat_indicators,
        "highlighted": highlighted,
        "ai_explanation": ai_explanation,
        "recommendation": get_safety_recommendation(scan.verdict),
    }
    return render(request, "detector/forensic_report.html", context)


@any_login_required
def forensic_report_pdf(request, scan_id):
    scan = get_scan_for_user(request, scan_id)

    urgency_found = urgency_flags(scan.email_text)
    financial_found = financial_flags(scan.email_text)
    urls = extract_urls(scan.email_text)
    url_flags, suspicious_url = url_threat_flags(urls)
    trusted_flag = is_trusted_domain(scan.sender_domain)
    unknown_pattern = unknown_sender_pattern(scan.sender_domain, trusted_flag)

    reasons = []
    if urgency_found: reasons.append("Urgency detected")
    if suspicious_url: reasons.append("Suspicious URL")
    if financial_found: reasons.append("Financial keywords")
    if unknown_pattern: reasons.append("Unknown sender pattern")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Title'], fontSize=18, textColor=colors.HexColor('#1e3a8a'))
    heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#1e293b'), spaceBefore=14, spaceAfter=6)
    normal_style = styles['Normal']

    elements = []
    elements.append(Paragraph("SecureMail AI - Forensic Report", title_style))
    elements.append(Spacer(1, 10))

    result_color = colors.HexColor('#b91c1c') if scan.result == "Phishing" else colors.HexColor('#15803d')
    result_style = ParagraphStyle('ResultStyle', parent=styles['Heading2'], textColor=result_color)
    elements.append(Paragraph(f"Classification: {scan.result.upper()}", result_style))
    elements.append(Spacer(1, 10))

    meta_data = [
        ["Scan ID", f"#{scan.id}"],
        ["Date & Time", scan.created_at.strftime("%d %b %Y, %H:%M")],
        ["Scanned by", scan.user.username if scan.user else "Unknown"],
        ["Detected Language", scan.language],
        ["Sender Domain", scan.sender_domain or "Unknown"],
        ["Risk Score", f"{scan.threat_score}/100"],
        ["Verdict", scan.verdict],
    ]
    meta_table = Table(meta_data, colWidths=[150, 300])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f1f5f9')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(meta_table)

    elements.append(Paragraph("Threat Indicators", heading_style))
    if reasons:
        for r in reasons:
            elements.append(Paragraph(f"• {r}", normal_style))
    else:
        elements.append(Paragraph("No threat indicators found.", normal_style))

    elements.append(Paragraph("AI Explanation", heading_style))
    explanation = (
        f"This email was classified as {scan.result} because the sender domain "
        f"({scan.sender_domain}) is {'verified and trusted' if trusted_flag else 'not recognized as trusted'}"
        f"{'; urgent/pressure language was detected' if urgency_found else ''}"
        f"{'; financial keywords were detected' if financial_found else ''}"
        f"{'; the email links to unverified domains' if suspicious_url else ''}."
    )
    elements.append(Paragraph(explanation, normal_style))

    elements.append(Paragraph("Safety Recommendation", heading_style))
    rec = get_safety_recommendation(scan.verdict)
    elements.append(Paragraph(rec["title"].replace("⚠ ", "").replace("✅ ", ""), normal_style))
    for point in rec["points"]:
        elements.append(Paragraph(f"• {point}", normal_style))

    doc.build(elements)
    buffer.seek(0)

    response = HttpResponse(buffer, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="forensic_report_scan_{scan.id}.pdf"'
    return response


# ======================= ADMIN area (no scanner) =======================

@admin_required
def admin_dashboard(request):
    total_users = User.objects.filter(is_staff=False).count()
    total_scans = ScanHistory.objects.count()

    today = timezone.now().date()
    trend_data = []
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        count = ScanHistory.objects.filter(created_at__date=day, result="Phishing").count()
        trend_data.append({"date": day.strftime("%d %b"), "count": count})

    lang_dist = []
    for lang in ["Tamil", "Malayalam", "Telugu", "English"]:
        count = ScanHistory.objects.filter(language=lang).count()
        pct = round((count / total_scans * 100), 1) if total_scans > 0 else 0
        lang_dist.append({"language": lang, "count": count, "pct": pct})

    risk_dist = []
    for risk in ["Safe", "Low Risk", "Medium Risk", "High Risk"]:
        count = ScanHistory.objects.filter(verdict=risk).count()
        pct = round((count / total_scans * 100), 1) if total_scans > 0 else 0
        risk_dist.append({"risk": risk, "count": count, "pct": pct})

    fp_qs = ScanHistory.objects.filter(reported_false_positive=True)
    false_positive_count = fp_qs.count()
    fp_reports = fp_qs.select_related('user').order_by('-created_at')[:10]

    model_performance = [
        {"model": "LSTM", "accuracy": "100%", "size": "15.27 MB", "speed": "0.74 ms"},
        {"model": "BiLSTM", "accuracy": "100%", "size": "15.87 MB", "speed": "1.03 ms"},
        {"model": "mBERT", "accuracy": "100%", "size": "681.27 MB", "speed": "177.73 ms"},
        {"model": "XLM-R", "accuracy": "100%", "size": "1076.99 MB", "speed": "150.21 ms"},
    ]

    context = {
        "total_users": total_users,
        "total_scans": total_scans,
        "trend_data": trend_data,
        "lang_dist": lang_dist,
        "risk_dist": risk_dist,
        "false_positive_count": false_positive_count,
        "fp_reports": fp_reports,
        "model_performance": model_performance,
    }
    return render(request, "detector/admin_dashboard.html", context)


@admin_required
def admin_history(request):
    selected_user = request.GET.get('user', '')
    qs = ScanHistory.objects.select_related('user').order_by('-created_at')
    if selected_user == '__none__':
        qs = qs.filter(user__isnull=True)
    elif selected_user:
        qs = qs.filter(user__username=selected_user)

    users = User.objects.filter(is_staff=False).order_by('username')
    return render(request, "detector/history.html", {
        "scans": qs[:100],
        "users": users,
        "selected_user": selected_user,
    })


@admin_required
def admin_analytics(request):
    qs = ScanHistory.objects.all()
    return render(request, "detector/analytics.html", build_analytics(qs, "All users"))