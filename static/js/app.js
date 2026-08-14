
const input = document.getElementById("imageInput");
const fileName = document.getElementById("fileName");
const preview = document.getElementById("preview");
const dropZone = document.getElementById("dropZone");

if (input) {
    input.addEventListener("change", function () {
        if (this.files.length) {
            const file = this.files[0];
            fileName.textContent = file.name;

            if (preview) {
                preview.src = URL.createObjectURL(file);
                preview.classList.add("show");
            }

            dropZone.classList.add("selected");
        }
    });
}

if (dropZone && input) {
    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, e => {
            e.preventDefault();
            dropZone.classList.add("dragging");
        });
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, e => {
            e.preventDefault();
            dropZone.classList.remove("dragging");
        });
    });

    dropZone.addEventListener("drop", e => {
        if (e.dataTransfer.files.length) {
            input.files = e.dataTransfer.files;
            input.dispatchEvent(new Event("change"));
        }
    });
}
