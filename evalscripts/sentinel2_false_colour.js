//VERSION=3
function setup() {
  return { input: ["B08", "B04", "B03", "dataMask"], output: { bands: 4 } };
}
function evaluatePixel(s) {
  return [2.5*s.B08, 2.5*s.B04, 2.5*s.B03, s.dataMask];
}
