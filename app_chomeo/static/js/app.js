let petFile = null;
let breedFile = null;

function scrollToApp() {
    document.getElementById("classifier").scrollIntoView({ behavior: "smooth" });
}

function scrollToHome() {
    document.getElementById("home").scrollIntoView({ behavior: "smooth" });
}

function switchTab(tabId) {
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.tab === tabId);
    });

    document.querySelectorAll(".tab-content").forEach(tab => {
        tab.classList.toggle("active", tab.id === tabId);
    });
}

function setupUpload(dropId, inputId, previewWrapId, previewId, setter) {
    const drop = document.getElementById(dropId);
    const input = document.getElementById(inputId);
    const previewWrap = document.getElementById(previewWrapId);
    const preview = document.getElementById(previewId);

    drop.addEventListener("click", () => input.click());

    drop.addEventListener("dragover", (e) => {
        e.preventDefault();
        drop.classList.add("drag-over");
    });

    drop.addEventListener("dragleave", () => {
        drop.classList.remove("drag-over");
    });

    drop.addEventListener("drop", (e) => {
        e.preventDefault();
        drop.classList.remove("drag-over");

        const file = e.dataTransfer.files[0];
        if (file) {
            handleFile(file, drop, previewWrap, preview, setter);
        }
    });

    input.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFile(file, drop, previewWrap, preview, setter);
        }
    });
}

function handleFile(file, drop, previewWrap, preview, setter) {
    const allowed = ["image/jpeg", "image/png", "image/webp", "image/bmp"];

    if (!allowed.includes(file.type)) {
        alert("Định dạng ảnh không được hỗ trợ. Vui lòng dùng JPG, PNG, WebP hoặc BMP.");
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        alert("Ảnh quá lớn. Vui lòng chọn ảnh dưới 10MB.");
        return;
    }

    setter(file);

    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        previewWrap.style.display = "flex";
        drop.style.display = "none";   // ẨN KHUNG KÉO THẢ SAU KHI CÓ ẢNH
    };
    reader.readAsDataURL(file);
}

function showLoading(targetId, text = "Đang xử lý ảnh...") {
    document.getElementById(targetId).innerHTML = `
        <div class="empty-state">
            <div class="loading">
                <span class="spinner"></span>
                <span>${text}</span>
            </div>
        </div>
    `;
}

function progressBar(percent) {
    const safe = Math.max(0, Math.min(100, percent));
    return `
        <div class="progress">
            <div class="progress-bar" style="width: ${safe}%"></div>
        </div>
    `;
}

function saveHistory(item) {
    const history = JSON.parse(sessionStorage.getItem("app_chomeo_history") || "[]");
    history.unshift({
        ...item,
        time: new Date().toLocaleString("vi-VN")
    });
    sessionStorage.setItem("app_chomeo_history", JSON.stringify(history.slice(0, 20)));
}

async function checkHealth() {
    const box = document.getElementById("healthBox");
    box.innerHTML = "Đang kiểm tra API...";

    try {
        const res = await fetch("/api/health");
        const data = await res.json();

        box.innerHTML = `
            <b>API:</b> ${data.success ? "Hoạt động" : "Lỗi"} |
            <b>Chó/mèo:</b> ${data.catdog_model_loaded ? "OK" : "Thiếu model"} |
            <b>Giống:</b> ${data.breed_model_loaded ? "OK" : "Thiếu model"} |
            <b>Số lớp giống:</b> ${data.class_count}
        `;
    } catch (err) {
        box.innerHTML = "Không kết nối được API.";
    }
}

async function classifyPet() {
    if (!petFile) {
        alert("Vui lòng tải ảnh trước.");
        return;
    }

    showLoading("petResult");

    const formData = new FormData();
    formData.append("image", petFile);

    try {
        const res = await fetch("/api/classify-pet", {
            method: "POST",
            body: formData
        });

        const data = await res.json();

        if (!data.success) {
            throw new Error(data.error || "Dự đoán thất bại.");
        }

        renderPetResult(data);
        saveHistory({ task: "catdog", result: data });

    } catch (err) {
        document.getElementById("petResult").innerHTML = `
            <div class="warning">Không nhận diện được. ${err.message}</div>
        `;
    }
}

function renderPetResult(data) {
    const target = document.getElementById("petResult");

    if (data.pet_type === "unknown") {
        target.innerHTML = `
            <div class="warning">
                Không nhận diện được. Vui lòng tải ảnh khác.
            </div>
            <div class="button-row">
                <button class="outline-btn small" onclick="resetPet()">Tải ảnh khác</button>
            </div>
        `;
        return;
    }

    const icon = data.pet_type === "dog" ? "🐶" : "🐱";
    const text = data.pet_type === "dog" ? "Đây là một con CHÓ" : "Đây là một con MÈO";
    const confidence = data.confidence * 100;
    const cat = data.prob_cat * 100;
    const dog = data.prob_dog * 100;

    target.innerHTML = `
        <div class="result-card">
            <div class="result-main">
                <div class="result-icon">${icon}</div>
                <div>
                    <h3 class="result-title">${text}</h3>
                    <div class="confidence">Độ tin cậy: <b>${confidence.toFixed(2)}%</b></div>
                </div>
            </div>

            <div class="breed-list">
                <div class="breed-row">
                    <div class="breed-row-top">
                        <span>🐱 Mèo</span>
                        <span>${cat.toFixed(2)}%</span>
                    </div>
                    ${progressBar(cat)}
                </div>

                <div class="breed-row">
                    <div class="breed-row-top">
                        <span>🐶 Chó</span>
                        <span>${dog.toFixed(2)}%</span>
                    </div>
                    ${progressBar(dog)}
                </div>
            </div>
        </div>
    `;
}

async function classifyBreed() {
    if (!breedFile) {
        alert("Vui lòng tải ảnh trước.");
        return;
    }

    showLoading("breedResult");

    const formData = new FormData();
    formData.append("image", breedFile);

    try {
        const res = await fetch("/api/classify-breed", {
            method: "POST",
            body: formData
        });

        const data = await res.json();

        if (!data.success) {
            throw new Error(data.error || "Dự đoán thất bại.");
        }

        renderBreedResult(data);
        saveHistory({ task: "breed", result: data });

    } catch (err) {
        document.getElementById("breedResult").innerHTML = `
            <div class="warning">Không nhận diện được. ${err.message}</div>
        `;
    }
}

function renderBreedResult(data) {
    const target = document.getElementById("breedResult");

    if (data.pet_type === "unknown") {
        target.innerHTML = `
            <div class="warning">
                Không phải chó hoặc mèo. Vui lòng tải ảnh khác.
            </div>
            <div class="button-row">
                <button class="outline-btn small" onclick="resetBreed()">Tải ảnh khác</button>
            </div>
        `;
        return;
    }

    const icon = data.pet_type === "dog" ? "🐶" : "🐱";
    const typeVi = data.pet_type === "dog" ? "Chó" : "Mèo";
    const confidence = data.confidence * 100;

    const topList = (data.top_5_breeds || []).map((item, idx) => {
        const p = item.confidence * 100;
        return `
            <div class="breed-row">
                <div class="breed-row-top">
                    <span>${idx + 1}. ${item.breed}</span>
                    <span>${p.toFixed(2)}%</span>
                </div>
                ${progressBar(p)}
            </div>
        `;
    }).join("");

    target.innerHTML = `
        <div class="result-card">
            <div class="result-main">
                <div class="result-icon">${icon}</div>
                <div>
                    <h3 class="result-title">${typeVi} - Giống: ${data.breed}</h3>
                    <div class="confidence">Độ tin cậy: <b>${confidence.toFixed(2)}%</b></div>
                </div>
            </div>

            <h3>Top 5 giống phù hợp nhất</h3>
            <div class="breed-list">
                ${topList}
            </div>
        </div>
    `;
}

function resetPet() {
    petFile = null;
    document.getElementById("petInput").value = "";
    document.getElementById("petPreviewWrap").style.display = "none";
    document.getElementById("petPreview").src = "";
    document.getElementById("dropPet").style.display = "block";   // hiện lại khung upload

    document.getElementById("petResult").innerHTML = `
        <div class="empty-state">
            <div>🐾</div>
            <p>Kết quả sẽ hiển thị tại đây.</p>
        </div>
    `;
}

function resetBreed() {
    breedFile = null;
    document.getElementById("breedInput").value = "";
    document.getElementById("breedPreviewWrap").style.display = "none";
    document.getElementById("breedPreview").src = "";
    document.getElementById("dropBreed").style.display = "block";   // hiện lại khung upload

    document.getElementById("breedResult").innerHTML = `
        <div class="empty-state">
            <div>🏷️</div>
            <p>Top 5 giống sẽ hiển thị tại đây.</p>
        </div>
    `;
}

document.addEventListener("DOMContentLoaded", () => {
    setupUpload(
        "dropPet",
        "petInput",
        "petPreviewWrap",
        "petPreview",
        (file) => { petFile = file; }
    );

    setupUpload(
        "dropBreed",
        "breedInput",
        "breedPreviewWrap",
        "breedPreview",
        (file) => { breedFile = file; }
    );
});
