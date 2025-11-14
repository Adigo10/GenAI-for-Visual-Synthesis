const form = document.getElementById("controlForm");
const imageInput = document.getElementById("imageInput");
const uploadZone = document.getElementById("uploadZone");
const browseButton = document.getElementById("browseButton");
const previewImage = document.getElementById("previewImage");
const statusChip = document.getElementById("statusChip");
const runButton = document.getElementById("runButton");
const toast = document.getElementById("toast");
const toastMessage = document.getElementById("toastMessage");
const stageGrid = document.getElementById("stageGrid");

let selectedFile = null;

const setStatus = (text, state = "idle") => {
	statusChip.textContent = text;
	statusChip.classList.remove("busy", "error");
	if (state === "busy") {
		statusChip.classList.add("busy");
	} else if (state === "error") {
		statusChip.classList.add("error");
	}
};

const showToast = (message, tone = "info", duration = 4000) => {
	toastMessage.textContent = message;
	toast.classList.remove("error");
	if (tone === "error") {
		toast.classList.add("error");
	}
	toast.hidden = false;
	clearTimeout(showToast.timer);
	showToast.timer = setTimeout(() => {
		toast.hidden = true;
	}, duration);
};

const getStageCard = (stageNumber) =>
	stageGrid.querySelector(`.stage-card[data-stage="${stageNumber}"]`);

const reorderStagesDescending = () => {
	const cards = Array.from(stageGrid.children);
	const sorted = cards.sort((a, b) => Number(b.dataset.stage) - Number(a.dataset.stage));
	sorted.forEach((card) => stageGrid.appendChild(card));
};

const setStageVisibility = (stageNumber, isVisible) => {
	const card = getStageCard(stageNumber);
	if (!card) return;
	card.hidden = !isVisible;
	if (isVisible) {
		reorderStagesDescending();
	}
};

const setStageLoading = (stageNumber, isLoading) => {
	const card = getStageCard(stageNumber);
	if (!card) return;
	const shell = card.querySelector(".image-shell");
	const img = shell.querySelector("img");
	const spinner = shell.querySelector(".spinner");

	if (isLoading) {
		shell.classList.add("loading");
		spinner.style.display = "block";
		img.hidden = true;
		img.src = "";
	} else {
		shell.classList.remove("loading");
		spinner.style.display = "none";
		img.hidden = false;
	}
};

const setStageImage = (stageNumber, base64Payload) => {
	const card = getStageCard(stageNumber);
	if (!card) return;
	setStageVisibility(stageNumber, true);
	const shell = card.querySelector(".image-shell");
	const img = shell.querySelector("img");
	img.src = `data:image/png;base64,${base64Payload}`;
	setStageLoading(stageNumber, false);
};

const resetStages = () => {
	[1, 2, 3, 4].forEach((stage) => {
		setStageVisibility(stage, false);
		setStageLoading(stage, true);
	});
	reorderStagesDescending();
};

const beginStage = (stageNumber) => {
	setStageVisibility(stageNumber, true);
	setStageLoading(stageNumber, true);
};

const readFilePreview = (file) => {
	const reader = new FileReader();
	reader.onload = (event) => {
		previewImage.src = event.target.result;
		previewImage.hidden = false;
	};
	reader.readAsDataURL(file);
};

const handleFileSelection = (file) => {
	if (!file) return;
	selectedFile = file;
	readFilePreview(file);
};

browseButton.addEventListener("click", () => imageInput.click());

uploadZone.addEventListener("dragover", (event) => {
	event.preventDefault();
	uploadZone.classList.add("dragover");
});

uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));

uploadZone.addEventListener("drop", (event) => {
	event.preventDefault();
	uploadZone.classList.remove("dragover");
	const [file] = event.dataTransfer.files;
	handleFileSelection(file);
});

imageInput.addEventListener("change", () => {
	const [file] = imageInput.files;
	handleFileSelection(file);
});

const postForm = async (url, formData) => {
	const response = await fetch(url, {
		method: "POST",
		body: formData,
	});

	if (!response.ok) {
		const errorPayload = await response.json().catch(() => ({}));
		const message = errorPayload.detail || response.statusText || "Request failed";
		throw new Error(message);
	}

	return response.json();
};

const runStage1 = (file) => {
	const data = new FormData();
	data.append("file", file, file.name || "upload.png");
	return postForm("/api/stage1", data);
};

const runStage2 = (runId, prompt, negativePrompt) => {
	const data = new FormData();
	data.append("run_id", runId);
	data.append("prompt", prompt);
	data.append("negative_prompt", negativePrompt);
	return postForm("/api/stage2", data);
};

const runStage3 = (runId) => {
	const data = new FormData();
	data.append("run_id", runId);
	return postForm("/api/stage3", data);
};

const runStage4 = (runId, prompt, negativePrompt) => {
	const data = new FormData();
	data.append("run_id", runId);
	data.append("prompt", prompt);
	data.append("negative_prompt", negativePrompt);
	return postForm("/api/stage4", data);
};

const runPipeline = async (event) => {
	event.preventDefault();

	if (!selectedFile) {
		showToast("Please choose an image before running the pipeline.", "error");
		setStatus("Waiting for image", "error");
		return;
	}

	resetStages();
	setStatus("Uploading image", "busy");
	runButton.disabled = true;

	try {
		const vehiclePrompt = form.vehiclePrompt.value.trim();
		const vehicleNegative = form.vehicleNegative.value;
		const backgroundPrompt = form.backgroundPrompt.value.trim();
		const backgroundNegative = form.backgroundNegative.value;

		if (!vehiclePrompt || !backgroundPrompt) {
			throw new Error("Both prompts are required to begin the pipeline.");
		}

		beginStage(1);
		const stage1 = await runStage1(selectedFile);
		setStageImage(1, stage1.image);
		setStatus("Rendering vehicle", "busy");

		beginStage(2);
		const stage2 = await runStage2(stage1.run_id, vehiclePrompt, vehicleNegative);
		setStageImage(2, stage2.image);
		setStatus("Analyzing edited vehicle", "busy");

		beginStage(3);
		const stage3 = await runStage3(stage1.run_id);
		setStageImage(3, stage3.image);
		setStatus("Inpainting background", "busy");

		beginStage(4);
		const stage4 = await runStage4(stage1.run_id, backgroundPrompt, backgroundNegative);
		setStageImage(4, stage4.image);
		reorderStagesDescending();

		setStatus("Complete", "idle");
		showToast("Pipeline finished successfully. Enjoy your remix!");
	} catch (error) {
		setStatus("Error", "error");
		showToast(error.message || "Pipeline failed", "error", 6000);
		console.error(error);
	} finally {
		runButton.disabled = false;
	}
};

form.addEventListener("submit", runPipeline);

// Ensure stage cards start in idle state
resetStages();
setStatus("Idle", "idle");
