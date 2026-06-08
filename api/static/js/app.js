const escapeHtml = value => String(value ?? "—").replace(/[&<>"']/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
}[char]));
const formatValue = value => value === null || value === undefined || value === "" ? "—" :
    (typeof value === "number" ? value.toLocaleString("pl-PL", { maximumFractionDigits: 4 }) : escapeHtml(value));
const setLoading = target => { target.innerHTML = '<div class="empty-state">Ładowanie danych…</div>'; };
const showError = (target, message) => { target.innerHTML = `<div class="alert alert-danger"><strong>Nie udało się pobrać danych.</strong><br>${escapeHtml(message)}</div>`; };

async function apiRequest(url, options = {}) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        const detail = Array.isArray(data.detail) ? data.detail.map(item => item.msg).join(", ") : data.detail;
        throw new Error(detail || `Błąd HTTP ${response.status}`);
    }
    if (data.error) throw new Error(data.error);
    return data;
}

function renderSummary(summary) {
    return `<div class="summary-grid">${Object.entries(summary || {}).map(([key, value]) =>
        `<div class="summary-item"><small>${escapeHtml(key.replaceAll("_", " "))}</small><strong>${formatValue(value)}</strong></div>`
    ).join("")}</div>`;
}

function renderTable(rows, columns) {
    if (!rows?.length) return '<div class="empty-state">Brak wyników.</div>';
    return `<div class="table-responsive"><table class="table table-hover align-middle">
        <thead><tr>${columns.map(column => `<th>${escapeHtml(column.label)}</th>`).join("")}</tr></thead>
        <tbody>${rows.map(row => `<tr>${columns.map(column => `<td>${formatValue(row[column.key])}</td>`).join("")}</tr>`).join("")}</tbody>
    </table></div>`;
}

const hotelColumns = [
    { key: "location_id", label: "Location ID" }, { key: "name", label: "Nazwa" }, { key: "city", label: "Miasto" },
    { key: "rating", label: "Ocena" }, { key: "price_level", label: "Cena" }, { key: "similarity_score", label: "Podobieństwo" }
];
const clusterHotelColumns = [
    { key: "location_id", label: "Location ID" }, { key: "name_details", label: "Nazwa" }, { key: "address_obj.city_details", label: "Miasto" },
    { key: "rating", label: "Ocena" }, { key: "num_reviews", label: "Opinie" }, { key: "price_level", label: "Cena" }
];

function renderCluster(cluster) {
    const hotels = cluster.examples || cluster.hotels || [];
    return `<article class="card result-card">
        <div class="card-header d-flex flex-wrap justify-content-between align-items-center gap-2"><h2 class="h5 mb-0">Klaster ${formatValue(cluster.cluster)}</h2><span class="badge text-bg-primary">${formatValue(cluster.count)} hoteli</span></div>
        <div class="card-body p-4"><p class="lead fs-6">${escapeHtml(cluster.cluster_info)}</p><h3 class="h6 mt-4">Średnie wartości cech</h3>${renderSummary(cluster.summary)}<h3 class="h6 mt-4">Hotele</h3>${renderTable(hotels, clusterHotelColumns)}</div>
    </article>`;
}

const clustersTarget = document.getElementById("clusters-result");
document.getElementById("load-clusters")?.addEventListener("click", async () => {
    setLoading(clustersTarget);
    try { clustersTarget.innerHTML = (await apiRequest("/clusters")).map(renderCluster).join(""); }
    catch (error) { showError(clustersTarget, error.message); }
});
document.getElementById("cluster-details-form")?.addEventListener("submit", async event => {
    event.preventDefault(); setLoading(clustersTarget); const form = new FormData(event.target);
    try { clustersTarget.innerHTML = renderCluster(await apiRequest(`/clusters/${form.get("cluster_id")}`)); }
    catch (error) { showError(clustersTarget, error.message); }
});

document.getElementById("recommend-form")?.addEventListener("submit", async event => {
    event.preventDefault(); const target = document.getElementById("recommend-result"); setLoading(target); const form = new FormData(event.target);
    try {
        const data = await apiRequest(`/recommend/${form.get("location_id")}?limit=${form.get("limit")}`);
        target.innerHTML = `<div class="source-hotel mb-3"><small class="text-secondary">Hotel źródłowy</small><h2 class="h5 mb-1">${escapeHtml(data.source_hotel.name)}</h2><span>${escapeHtml(data.source_hotel.city)} · ID ${formatValue(data.source_hotel.location_id)}</span></div><div class="card result-card"><div class="card-header"><h2 class="h5 mb-0">Rekomendowane hotele</h2></div>${renderTable(data.recommendations, hotelColumns)}</div>`;
    } catch (error) { showError(target, error.message); }
});
document.getElementById("search-form")?.addEventListener("submit", async event => {
    event.preventDefault(); const target = document.getElementById("search-result"); setLoading(target); const form = new FormData(event.target);
    try {
        const data = await apiRequest(`/search-hotels?query=${encodeURIComponent(form.get("query"))}&limit=${form.get("limit")}`);
        target.innerHTML = `<div class="card result-card"><div class="card-header"><h2 class="h5 mb-0">Wyniki dla „${escapeHtml(data.query)}”</h2></div>${renderTable(data.results, hotelColumns)}</div>`;
    } catch (error) { showError(target, error.message); }
});
document.getElementById("sentiment-form")?.addEventListener("submit", async event => {
    event.preventDefault(); const target = document.getElementById("sentiment-result"); setLoading(target); const form = new FormData(event.target);
    try {
        const data = await apiRequest("/predict-sentiment", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ review: form.get("review") }) });
        const positive = data.sentiment === "positive";
        const probabilities = data.probability ? renderSummary({ "prawdopodobieństwo negatywne": data.probability.negative, "prawdopodobieństwo pozytywne": data.probability.positive }) : "";
        target.innerHTML = `<div class="sentiment-result ${positive ? "sentiment-positive" : "sentiment-negative"}"><span class="badge ${positive ? "text-bg-success" : "text-bg-danger"}">Etykieta ${formatValue(data.label)}</span><h2 class="mt-3">${positive ? "Pozytywna opinia" : "Negatywna opinia"}</h2><p>${escapeHtml(data.review)}</p>${probabilities}</div>`;
    } catch (error) { showError(target, error.message); }
});

const optionalNumber = (form, name) => form.get(name) === "" ? null : Number(form.get(name));
document.getElementById("predict-cluster-form")?.addEventListener("submit", async event => {
    event.preventDefault(); const target = document.getElementById("predict-cluster-result"); setLoading(target); const form = new FormData(event.target);
    const numericFields = ["rating", "num_reviews", "ranking", "ranking_out_of", "location_rating", "rooms_rating", "service_rating", "value_rating", "cleanliness_rating", "review_rating_count_1", "review_rating_count_2", "review_rating_count_3", "review_rating_count_4", "review_rating_count_5"];
    const payload = { name_details: form.get("name_details"), price_level: form.get("price_level") || null };
    numericFields.forEach(name => { payload[name] = optionalNumber(form, name); });
    try {
        const data = await apiRequest("/predict-cluster", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
        const warning = data.prediction_warning ? `<div class="alert alert-warning mt-3 mb-0">${escapeHtml(data.prediction_warning)}</div>` : "";
        const features = data.imputed_features?.length ? data.imputed_features.map(item => `<span class="badge text-bg-light border me-1 mb-1">${escapeHtml(item)}</span>`).join("") : "Brak";
        target.innerHTML = `<div class="card result-card"><div class="card-header"><h2 class="h5 mb-0">${escapeHtml(data.hotel)}</h2></div><div class="card-body p-4"><div class="display-6 fw-bold text-primary">Klaster ${formatValue(data.cluster)}</div><p class="lead fs-6 mt-2">${escapeHtml(data.cluster_info)}</p><h3 class="h6 mt-4">Imputowane cechy: ${formatValue(data.imputed_count)}</h3><div>${features}</div>${warning}</div></div>`;
    } catch (error) { showError(target, error.message); }
});
