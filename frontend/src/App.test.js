import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";

const RESULT = {
  label: "Psoriasis",
  is_psoriasis: true,
  probability_psoriasis: 0.93,
  confidence: 0.93,
  threshold: 0.5,
  heatmap: "data:image/png;base64,AAAA",
  model: { backend: "torch", name: "ResNet18 transfer learning (PyTorch)" },
};

const photo = () => new File(["pixels"], "patch.jpg", { type: "image/jpeg" });

beforeEach(() => {
  global.URL.createObjectURL = jest.fn(() => "blob:preview");
  global.URL.revokeObjectURL = jest.fn();
  global.fetch = jest.fn();
});

test("shows the disclaimer and keeps Analyze disabled until a photo is chosen", () => {
  render(<App />);
  expect(screen.getByText(/not a medical device/i)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /analyze photo/i })).toBeDisabled();
});

test("rejects files that are not supported images", () => {
  render(<App />);
  userEvent.upload(screen.getByLabelText(/choose a photo/i), new File(["x"], "notes.txt", { type: "text/plain" }));
  expect(screen.getByRole("alert")).toHaveTextContent(/jpeg, png, webp or bmp/i);
  expect(screen.getByRole("button", { name: /analyze photo/i })).toBeDisabled();
});

test("sends the photo to the API and shows the result", async () => {
  global.fetch.mockResolvedValue({ ok: true, json: async () => RESULT });
  render(<App />);
  userEvent.upload(screen.getByLabelText(/choose a photo/i), photo());
  userEvent.click(screen.getByRole("button", { name: /analyze photo/i }));

  expect(await screen.findByText("Psoriasis")).toBeInTheDocument();
  expect(screen.getByText(/probability 93%/i)).toBeInTheDocument();
  expect(screen.getByAltText(/heatmap/i)).toHaveAttribute("src", RESULT.heatmap);
  const [url, options] = global.fetch.mock.calls[0];
  expect(url).toMatch(/\/predict$/);
  expect(options.method).toBe("POST");
  expect(options.body.get("image").name).toBe("patch.jpg");
});

test("shows the server's error message", async () => {
  global.fetch.mockResolvedValue({
    ok: false,
    status: 415,
    json: async () => ({ error: { code: "invalid_image", message: "The file is not a readable image." } }),
  });
  render(<App />);
  userEvent.upload(screen.getByLabelText(/choose a photo/i), photo());
  userEvent.click(screen.getByRole("button", { name: /analyze photo/i }));
  await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("The file is not a readable image."));
});

test("shows a friendly message when the server is unreachable", async () => {
  global.fetch.mockRejectedValue(new TypeError("Failed to fetch"));
  render(<App />);
  userEvent.upload(screen.getByLabelText(/choose a photo/i), photo());
  userEvent.click(screen.getByRole("button", { name: /analyze photo/i }));
  await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/could not reach the analysis server/i));
});
