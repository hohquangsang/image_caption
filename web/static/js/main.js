/* ═══════════════════════════════════════════════════════
   VisioCaption AI - main.js
   Xử lý UI: tab switching, file upload, API call, etc.
═══════════════════════════════════════════════════════ */

// ─── Eye Care Mode ───────────────────────────────────────────
(function initEyeCare() {
  if (localStorage.getItem('eyeCare') === '1') {
    document.body.classList.add('eye-care');
  }
})();

function toggleEyeCare() {
  const on = document.body.classList.toggle('eye-care');
  localStorage.setItem('eyeCare', on ? '1' : '0');
}

// ─── Hamburger Menu ───────────────────────────────────────────
function toggleMenu() {
  const links = document.getElementById('navLinks');
  const btn = document.getElementById('hamburger');
  links.classList.toggle('open');
  btn.classList.toggle('active');
}

// ─── Particle Background ─────────────────────────────────────
(function initParticles() {
  const container = document.getElementById("particles");
  if (!container) return;
  const count = 20;
  for (let i = 0; i < count; i++) {
    const p = document.createElement("div");
    p.className = "particle";
    const size = Math.random() * 4 + 2;
    p.style.cssText = `
      width:${size}px; height:${size}px;
      left:${Math.random() * 100}%;
      animation-duration:${Math.random() * 15 + 10}s;
      animation-delay:${Math.random() * -15}s;
      opacity:${Math.random() * 0.5 + 0.1};
    `;
    container.appendChild(p);
  }
})();

// ─── Navbar scroll effect ─────────────────────────────────────
window.addEventListener("scroll", () => {
  const navbar = document.getElementById("navbar");
  if (window.scrollY > 40) navbar.classList.add("scrolled");
  else navbar.classList.remove("scrolled");
});

// ─── Tab Switching (Demo Section) ────────────────────────────
function switchTab(tab) {
  // Update buttons
  document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
  document.getElementById(`tab-${tab}`).classList.add("active");
  // Update contents
  document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
  document.getElementById(`content-${tab}`).classList.add("active");
}

// ─── Guide Tab Switch ─────────────────────────────────────────
function switchGuide(tab) {
  document.querySelectorAll(".guide-tab").forEach(btn => btn.classList.remove("active"));
  event.target.classList.add("active");

  const img2cap = document.getElementById("guide-img2cap");
  const cap2img = document.getElementById("guide-cap2img");
  if (tab === "img2cap") {
    img2cap.style.display = "flex";
    cap2img.style.display = "none";
  } else {
    img2cap.style.display = "none";
    cap2img.style.display = "block";
  }
}

// ─── File Upload ──────────────────────────────────────────────
let currentFile = null;

function handleFileSelect(event) {
  const file = event.target.files[0];
  if (file) loadFile(file);
}

function handleDrop(event) {
  event.preventDefault();
  const zone = document.getElementById("uploadZone");
  zone.classList.remove("drag-active");
  const file = event.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) loadFile(file);
  else showToast("Vui lòng chọn file ảnh", "error");
}

function handleDragOver(event) {
  event.preventDefault();
  document.getElementById("uploadZone").classList.add("drag-active");
}

function handleDragLeave() {
  document.getElementById("uploadZone").classList.remove("drag-active");
}

function loadFile(file) {
  // Kiểm tra kích thước
  if (file.size > 10 * 1024 * 1024) {
    showToast("File ảnh quá lớn! Tối đa 10MB", "error");
    return;
  }
  currentFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById("imagePreview").src = e.target.result;
    document.getElementById("uploadZone").style.display = "none";
    document.getElementById("previewContainer").style.display = "flex";
    // Reset results
    resetResults();
  };
  reader.readAsDataURL(file);
}

function resetUpload() {
  currentFile = null;
  document.getElementById("fileInput").value = "";
  document.getElementById("uploadZone").style.display = "flex";
  document.getElementById("previewContainer").style.display = "none";
  resetResults();
}

function resetResults() {
  document.getElementById("resultsEmpty").style.display = "flex";
  document.getElementById("captionsList").style.display = "none";
  document.getElementById("loadingState").style.display = "none";
  document.getElementById("errorState").style.display = "none";
  document.getElementById("copyAllBtn").style.display = "none";
  document.getElementById("resultsCount").textContent = "0/5";
}

// ─── Generate Captions ───────────────────────────────────────
async function generateCaptions() {
  if (!currentFile) {
    showToast("Vui lòng chọn ảnh trước!", "error");
    return;
  }

  // UI: Loading state
  setLoadingState(true);

  try {
    const formData = new FormData();
    formData.append("image", currentFile);

    const response = await fetch("/api/generate-captions", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      throw new Error(data.error || "Lỗi không xác định");
    }

    // Show captions
    displayCaptions(data.captions);
    showToast(`✅ Đã sinh ${data.captions.length} captions!`, "success");

  } catch (error) {
    console.error("Error:", error);
    showError(error.message || "Không thể kết nối đến server. Hãy chắc chắn server đang chạy.");
  } finally {
    setLoadingState(false);
  }
}

function setLoadingState(loading) {
  const generateBtn = document.getElementById("generateBtn");
  const generateBtnText = document.getElementById("generateBtnText");
  const spinner = document.getElementById("spinner");
  const loadingState = document.getElementById("loadingState");
  const resultsEmpty = document.getElementById("resultsEmpty");

  if (loading) {
    generateBtn.disabled = true;
    generateBtnText.textContent = "Đang xử lý...";
    spinner.style.display = "block";
    loadingState.style.display = "flex";
    resultsEmpty.style.display = "none";
    document.getElementById("captionsList").style.display = "none";
    document.getElementById("errorState").style.display = "none";
    document.getElementById("copyAllBtn").style.display = "none";
  } else {
    generateBtn.disabled = false;
    generateBtnText.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg> Sinh Caption`;
    spinner.style.display = "none";
    loadingState.style.display = "none";
  }
}

function displayCaptions(captions) {
  const list = document.getElementById("captionsList");
  const empty = document.getElementById("resultsEmpty");
  const copyAllBtn = document.getElementById("copyAllBtn");
  const countEl = document.getElementById("resultsCount");

  list.innerHTML = "";
  captions.forEach((caption, idx) => {
    const item = document.createElement("div");
    item.className = "caption-item";
    item.style.animationDelay = `${idx * 0.06}s`;
    item.innerHTML = `
      <div class="caption-num">${idx + 1}</div>
      <div class="caption-text">${escapeHtml(caption)}</div>
      <button class="caption-copy" onclick="copyCaption('${escapeHtml(caption)}')" title="Sao chép">
        📋
      </button>
    `;
    list.appendChild(item);
  });

  empty.style.display = "none";
  list.style.display = "flex";
  copyAllBtn.style.display = "block";
  countEl.textContent = `${captions.length}/5`;
}

function showError(msg) {
  document.getElementById("resultsEmpty").style.display = "none";
  document.getElementById("captionsList").style.display = "none";
  document.getElementById("errorState").style.display = "flex";
  document.getElementById("errorMsg").textContent = msg;
}

// ─── Copy Functions ───────────────────────────────────────────
function copyCaption(text) {
  navigator.clipboard.writeText(text).then(() => {
    showToast("Đã sao chép!", "success");
  }).catch(() => {
    // Fallback
    const el = document.createElement("textarea");
    el.value = text;
    document.body.appendChild(el);
    el.select();
    document.execCommand("copy");
    document.body.removeChild(el);
    showToast("Đã sao chép!", "success");
  });
}

function copyAllCaptions() {
  const items = document.querySelectorAll(".caption-text");
  const texts = Array.from(items).map((el, i) => `${i + 1}. ${el.textContent}`).join("\n");
  copyCaption(texts);
}

// ─── Toast Notification ───────────────────────────────────────
let toastTimeout;
function showToast(message, type = "info") {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.className = `toast ${type} show`;
  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => {
    toast.classList.remove("show");
  }, 3000);
}

// ─── Helper: Escape HTML ──────────────────────────────────────
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// ─── Intersection Observer (animations) ──────────────────────
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.opacity = "1";
      entry.target.style.transform = "translateY(0)";
    }
  });
}, { threshold: 0.1 });

document.querySelectorAll(".pipe-step, .feature-card, .info-card").forEach(el => {
  el.style.opacity = "0";
  el.style.transform = "translateY(20px)";
  el.style.transition = "opacity 0.6s ease, transform 0.6s ease";
  observer.observe(el);
});

// ─── Prevent default drop on body ────────────────────────────
document.body.addEventListener("dragover", (e) => e.preventDefault());
document.body.addEventListener("drop", (e) => e.preventDefault());
