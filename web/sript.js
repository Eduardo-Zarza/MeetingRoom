const dropArea  = document.getElementById("drop-area");
const inputFile = document.getElementById("input-file");
const imageView = document.getElementById("img-view");
let currentImageSrc = null;

inputFile.addEventListener("change", handleFile);

async function handleFile() {
    const file = inputFile.files[0];
    if (!file) return;

    // mostrar la imagen como fondo (opcional)
    const imgLink = URL.createObjectURL(file);
    currentImageSrc = imgLink;
    imageView.style.backgroundImage = `url(${imgLink})`;
    imageView.style.backgroundSize = "cover";
    imageView.style.backgroundPosition = "center";

    // estado "cargando"
    imageView.classList.remove("estado-ok", "estado-error");
    imageView.classList.add("estado-loading");
    imageView.innerHTML = "<p>Cargando...</p>";

    const resultado = await detectarSala(file);

    // con la respuesta, actualizamos la vista
    actualizarVista(resultado);
}

// Función real: llama a la API local de FastAPI
async function detectarSala(file) {
    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            console.error("Error HTTP:", response.status, response.statusText);
            return {
                status: "error",
                issues: ["Error en el servidor (" + response.status + ")"]
            };
        }

        const data = await response.json();
        // La API devuelve algo como { status: "ok" } o { status: "issues", issues: [...] }
        return data;
    } catch (err) {
        console.error("Error al llamar a la API:", err);
        return {
            status: "error",
            issues: ["No se pudo conectar al servidor"]
        };
    }
}

function actualizarVista(res) {
    imageView.classList.remove("estado-loading");
    imageView.className = "resultado"; // base
    // quitar la foto de fondo para usar solo el layout de resultados
    imageView.style.backgroundImage = "none";
    imageView.style.backgroundSize = "";
    imageView.style.backgroundPosition = "";

    if (res.status === "ok") {
        imageView.classList.add("estado-ok");
        imageView.innerHTML = `
    <div class="resultado-contenido resultado-ok">
        <div class="col-foto">
            <img src="images/check.png" alt="Todo correcto" class="check-icon">
        </div>
        <div class="col-texto">
            <p>Todo está en orden.</p>
            <p>¡Muchas gracias!</p>
            <button id="btn-volver">Subir otra imagen</button>
        </div>
    </div>
`;
        document
            .getElementById("btn-volver")
            .addEventListener("click", resetVista);
    } else if (res.status === "issues") {
        imageView.classList.add("estado-error");

        let lista = "";
        (res.issues || []).forEach(item => {
            const limpio = String(item).replace(/â€¢/g, "").trim();
            lista += `<li>${limpio}</li>`;
        });

        imageView.innerHTML = `
            <div class="resultado-issues">
                <p>Se detectaron los siguientes detalles:</p>
                <div class="resultado-issues-body">
                    <div class="col-foto">
                        ${currentImageSrc ? `<img src="${currentImageSrc}" alt="Foto de la sala">` : ""}
                    </div>
                    <div class="col-texto">
                        <ul>${lista}</ul>
                        <button id="btn-volver">Subir otra imagen</button>
                    </div>
                </div>
            </div>
        `;

        document
            .getElementById("btn-volver")
            .addEventListener("click", resetVista);
    } else {
        // Estado de error genérico
        imageView.classList.add("estado-error");
        const mensaje = (res.issues && res.issues.length)
            ? res.issues.join("<br>")
            : "Ocurrió un error al analizar la sala.";

        imageView.innerHTML = `
            <p>${mensaje}</p>
            <button id="btn-volver">Subir otra imagen</button>
        `;

        document
            .getElementById("btn-volver")
            .addEventListener("click", resetVista);
    }
}

function resetVista() {
    imageView.style.backgroundImage = "none";
    imageView.className = "resultado";
    imageView.innerHTML = `
        <img src="images/upload.png">
        <p>Arrastra o da click aquí <br> para subir una imagen</p>
        <span>Sube una imagen desde tu dispositivo</span>
    `;
}
