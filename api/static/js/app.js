function escapeHtml(value) {
    if (value === null || value === undefined) {
        return "—";
    }

    return String(value).replace(/[&<>"']/g, function (character) {
        const replacements = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#039;"
        };

        return replacements[character];
    });
}

function formatValue(value) {
    if (value === null || value === undefined || value === "") {
        return "—";
    }

    if (typeof value === "number") {
        return value.toLocaleString("pl-PL", { maximumFractionDigits: 4 });
    }

    return escapeHtml(value);
}

function showLoading(target) {
    target.innerHTML = '<div class="empty-state">Ładowanie danych…</div>';
}

function showError(target, message) {
    target.innerHTML = `
        <div class="alert alert-danger">
            <strong>Nie udało się pobrać danych.</strong><br>
            ${escapeHtml(message)}
        </div>
    `;
}

async function apiRequest(url, options) {
    const response = await fetch(url, options);
    const data = await response.json();

    if (!response.ok) {
        throw new Error("Błąd HTTP " + response.status);
    }

    if (data.error) {
        throw new Error(data.error);
    }

    return data;
}

function renderSummary(summary) {
    let items = "";

    for (const key in summary) {
        const label = key.replaceAll("_", " ");

        items += `
            <div class="summary-item">
                <small>${escapeHtml(label)}</small>
                <strong>${formatValue(summary[key])}</strong>
            </div>
        `;
    }

    return `<div class="summary-grid">${items}</div>`;
}

function renderTable(rows, columns) {
    if (!rows || rows.length === 0) {
        return '<div class="empty-state">Brak wyników.</div>';
    }

    let headers = "";
    let body = "";

    for (const column of columns) {
        headers += `<th>${escapeHtml(column.label)}</th>`;
    }

    for (const row of rows) {
        let cells = "";

        for (const column of columns) {
            cells += `<td>${formatValue(row[column.key])}</td>`;
        }

        body += `<tr>${cells}</tr>`;
    }

    return `
        <div class="table-responsive">
            <table class="table table-hover align-middle">
                <thead><tr>${headers}</tr></thead>
                <tbody>${body}</tbody>
            </table>
        </div>
    `;
}

const hotelColumns = [
    { key: "location_id", label: "Location ID" },
    { key: "name", label: "Nazwa" },
    { key: "city", label: "Miasto" },
    { key: "rating", label: "Ocena" },
    { key: "price_level", label: "Cena" },
    { key: "similarity_score", label: "Podobieństwo" }
];

const clusterHotelColumns = [
    { key: "location_id", label: "Location ID" },
    { key: "name_details", label: "Nazwa" },
    { key: "address_obj.city_details", label: "Miasto" },
    { key: "rating", label: "Ocena" },
    { key: "num_reviews", label: "Opinie" },
    { key: "price_level", label: "Cena" }
];

function renderCluster(cluster) {
    const hotels = cluster.examples || cluster.hotels || [];

    return `
        <article class="card result-card">
            <div class="card-header d-flex flex-wrap justify-content-between align-items-center gap-2">
                <h2 class="h5 mb-0">Klaster ${formatValue(cluster.cluster)}</h2>
                <span class="badge text-bg-primary">${formatValue(cluster.count)} hoteli</span>
            </div>
            <div class="card-body p-4">
                <p class="lead fs-6">${escapeHtml(cluster.cluster_info)}</p>
                <h3 class="h6 mt-4">Średnie wartości cech</h3>
                ${renderSummary(cluster.summary)}
                <h3 class="h6 mt-4">Hotele</h3>
                ${renderTable(hotels, clusterHotelColumns)}
            </div>
        </article>
    `;
}

const loadClustersButton = document.getElementById("load-clusters");

if (loadClustersButton) {
    loadClustersButton.addEventListener("click", async function () {
        const target = document.getElementById("clusters-result");
        showLoading(target);

        try {
            const clusters = await apiRequest("/clusters");
            let result = "";

            for (const cluster of clusters) {
                result += renderCluster(cluster);
            }

            target.innerHTML = result;
        } catch (error) {
            showError(target, error.message);
        }
    });
}

const clusterForm = document.getElementById("cluster-details-form");

if (clusterForm) {
    clusterForm.addEventListener("submit", async function (event) {
        event.preventDefault();

        const target = document.getElementById("clusters-result");
        const formData = new FormData(clusterForm);
        const clusterId = formData.get("cluster_id");

        showLoading(target);

        try {
            const cluster = await apiRequest("/clusters/" + clusterId);
            target.innerHTML = renderCluster(cluster);
        } catch (error) {
            showError(target, error.message);
        }
    });
}

const recommendForm = document.getElementById("recommend-form");

if (recommendForm) {
    recommendForm.addEventListener("submit", async function (event) {
        event.preventDefault();

        const target = document.getElementById("recommend-result");
        const formData = new FormData(recommendForm);
        const locationId = formData.get("location_id");
        const limit = formData.get("limit");

        showLoading(target);

        try {
            const data = await apiRequest("/recommend/" + locationId + "?limit=" + limit);

            target.innerHTML = `
                <div class="source-hotel mb-3">
                    <small class="text-secondary">Hotel źródłowy</small>
                    <h2 class="h5 mb-1">${escapeHtml(data.source_hotel.name)}</h2>
                    <span>${escapeHtml(data.source_hotel.city)} · ID ${formatValue(data.source_hotel.location_id)}</span>
                </div>
                <div class="card result-card">
                    <div class="card-header"><h2 class="h5 mb-0">Rekomendowane hotele</h2></div>
                    ${renderTable(data.recommendations, hotelColumns)}
                </div>
            `;
        } catch (error) {
            showError(target, error.message);
        }
    });
}

const searchForm = document.getElementById("search-form");

if (searchForm) {
    searchForm.addEventListener("submit", async function (event) {
        event.preventDefault();

        const target = document.getElementById("search-result");
        const formData = new FormData(searchForm);
        const query = encodeURIComponent(formData.get("query"));
        const limit = formData.get("limit");

        showLoading(target);

        try {
            const data = await apiRequest("/search-hotels?query=" + query + "&limit=" + limit);

            target.innerHTML = `
                <div class="card result-card">
                    <div class="card-header"><h2 class="h5 mb-0">Wyniki dla „${escapeHtml(data.query)}”</h2></div>
                    ${renderTable(data.results, hotelColumns)}
                </div>
            `;
        } catch (error) {
            showError(target, error.message);
        }
    });
}

const sentimentForm = document.getElementById("sentiment-form");

if (sentimentForm) {
    sentimentForm.addEventListener("submit", async function (event) {
        event.preventDefault();

        const target = document.getElementById("sentiment-result");
        const formData = new FormData(sentimentForm);

        showLoading(target);

        try {
            const data = await apiRequest("/predict-sentiment", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ review: formData.get("review") })
            });

            const isPositive = data.sentiment === "positive";
            const resultClass = isPositive ? "sentiment-positive" : "sentiment-negative";
            const badgeClass = isPositive ? "text-bg-success" : "text-bg-danger";
            const resultTitle = isPositive ? "Pozytywna opinia" : "Negatywna opinia";
            let probabilities = "";

            if (data.probability) {
                probabilities = renderSummary({
                    "prawdopodobieństwo negatywne": data.probability.negative,
                    "prawdopodobieństwo pozytywne": data.probability.positive
                });
            }

            target.innerHTML = `
                <div class="sentiment-result ${resultClass}">
                    <span class="badge ${badgeClass}">Etykieta ${formatValue(data.label)}</span>
                    <h2 class="mt-3">${resultTitle}</h2>
                    <p>${escapeHtml(data.review)}</p>
                    ${probabilities}
                </div>
            `;
        } catch (error) {
            showError(target, error.message);
        }
    });
}

function optionalNumber(formData, fieldName) {
    const value = formData.get(fieldName);

    if (value === "") {
        return null;
    }

    return Number(value);
}

const predictClusterForm = document.getElementById("predict-cluster-form");

if (predictClusterForm) {
    predictClusterForm.addEventListener("submit", async function (event) {
        event.preventDefault();

        const target = document.getElementById("predict-cluster-result");
        const formData = new FormData(predictClusterForm);
        const numericFields = [
            "rating",
            "num_reviews",
            "ranking",
            "ranking_out_of",
            "location_rating",
            "rooms_rating",
            "service_rating",
            "value_rating",
            "cleanliness_rating",
            "review_rating_count_1",
            "review_rating_count_2",
            "review_rating_count_3",
            "review_rating_count_4",
            "review_rating_count_5"
        ];

        const payload = {
            name_details: formData.get("name_details"),
            price_level: formData.get("price_level") || null
        };

        for (const fieldName of numericFields) {
            payload[fieldName] = optionalNumber(formData, fieldName);
        }

        showLoading(target);

        try {
            const data = await apiRequest("/predict-cluster", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            let warning = "";
            let imputedFeatures = "Brak";

            if (data.prediction_warning) {
                warning = `<div class="alert alert-warning mt-3 mb-0">${escapeHtml(data.prediction_warning)}</div>`;
            }

            if (data.imputed_features && data.imputed_features.length > 0) {
                imputedFeatures = "";

                for (const feature of data.imputed_features) {
                    imputedFeatures += `<span class="badge text-bg-light border me-1 mb-1">${escapeHtml(feature)}</span>`;
                }
            }

            target.innerHTML = `
                <div class="card result-card">
                    <div class="card-header"><h2 class="h5 mb-0">${escapeHtml(data.hotel)}</h2></div>
                    <div class="card-body p-4">
                        <div class="display-6 fw-bold text-primary">Klaster ${formatValue(data.cluster)}</div>
                        <p class="lead fs-6 mt-2">${escapeHtml(data.cluster_info)}</p>
                        <h3 class="h6 mt-4">Imputowane cechy: ${formatValue(data.imputed_count)}</h3>
                        <div>${imputedFeatures}</div>
                        ${warning}
                    </div>
                </div>
            `;
        } catch (error) {
            showError(target, error.message);
        }
    });
}
