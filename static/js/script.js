document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("file-upload");
    const fileLabel = document.querySelector(".custom-file-upload");

    // Update the label text when a file is selected
    fileInput.addEventListener("change", function () {
        const fileName = fileInput.files[0] ? fileInput.files[0].name : "Choose File";
        fileLabel.textContent = fileName;
    });
});
