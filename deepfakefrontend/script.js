// script.js
document.addEventListener("DOMContentLoaded", function () {
  const fileInput = document.getElementById("file-input");
  const imagePreview = document.getElementById("image-preview");
  const predictButton = document.getElementById("predict-button");
  const resultSection = document.getElementById("result-section");

  fileInput.addEventListener("change", function () {
    const file = fileInput.files[0];

    if (file) {
      const reader = new FileReader();

      reader.onload = function (e) {
        imagePreview.src = e.target.result;
        imagePreview.style.display = "block"; // Show the image
      };

      reader.readAsDataURL(file);
    } else {
      imagePreview.src = "#";
      imagePreview.style.display = "none"; // Hide when no image
    }
  });

  predictButton.addEventListener("click", function () {
    // This is where your client-side deepfake detection would go
    //  IMPORTANT:  Without server-side processing or a client-side
    //  deep learning library, you CANNOT perform accurate deepfake detection.
    //  This example just generates a random result.

    const randomResult = Math.random() < 0.5 ? "real" : "fake";

    if (randomResult === "real") {
      resultSection.innerHTML =
        '<p class="real-result"><span role="img" aria-label="check">✔️</span> Image is Real</p>';
    } else {
      resultSection.innerHTML =
        '<p class="fake-result"><span role="img" aria-label="cross">❌</span> Image is Fake</p>';
    }
  });
});
