const params = new URLSearchParams(location.search);
const sourceURIField = document.getElementById("sourceTabURI")!;
const url = params.get("url");
sourceURIField.textContent = url ?? "(unknown URL)";

// Tell TS this is an ES module.
export {};
