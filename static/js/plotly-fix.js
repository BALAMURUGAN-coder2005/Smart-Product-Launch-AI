// FIX 1: Force Plotly to resize after page load
window.addEventListener("load", function () {
    let graphs = document.querySelectorAll(".chart-box iframe");

    graphs.forEach((g) => {
        try {
            g.contentWindow.dispatchEvent(new Event("resize"));
        } catch (e) {
            console.warn("Resize event skipped:", e);
        }
    });
});

// FIX 2: Ensure rendering works inside hidden containers or cards
document.addEventListener("DOMContentLoaded", function () {
    setTimeout(() => {
        let plotDivs = document.querySelectorAll(".chart-box div.js-plotly-plot");
        plotDivs.forEach((div) => {
            Plotly.Plots.resize(div);
        });
    }, 300);
});

// FIX 3: Re-render charts on window resize
window.addEventListener("resize", () => {
    let plotDivs = document.querySelectorAll(".chart-box div.js-plotly-plot");
    plotDivs.forEach((div) => {
        Plotly.Plots.resize(div);
    });
});
