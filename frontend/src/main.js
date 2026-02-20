async function uploadImage() {
    const fileInput = document.getElementById("imageInput");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select an image.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("http://localhost:8000/diagnose", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        document.getElementById("prediction").textContent =
            JSON.stringify(data.prediction, null, 2);

        document.getElementById("report").textContent =
            data.report;

        document.getElementById("heatmap").src =
            "data:image/jpeg;base64," + data.heatmap_base64;

        document.getElementById("resultSection").style.display = "block";

    } catch (error) {
        console.error(error);
        alert("Error connecting to backend.");
    }
}

window.uploadImage = uploadImage;
