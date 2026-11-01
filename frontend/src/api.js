export const API_URL = (process.env.REACT_APP_API_URL || "http://localhost:5000").replace(/\/$/, "");

export class ApiError extends Error {}

/** Send one image to the PsoriScan API and return its JSON result. */
export async function analyzeImage(file, signal) {
  const body = new FormData();
  body.append("image", file);

  let response;
  try {
    response = await fetch(`${API_URL}/predict`, { method: "POST", body, signal });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new ApiError("Could not reach the analysis server. Try again in a moment.");
  }

  let data = null;
  try {
    data = await response.json();
  } catch {
    // non-JSON error body; handled below
  }
  if (!response.ok) {
    throw new ApiError(data?.error?.message || `The server returned an error (${response.status}).`);
  }
  return data;
}
