//VERSION=3
function setup() {
  return { input: ["B04", "B03", "B02", "dataMask"], output: { bands: 4 } };
}
function evaluatePixel(s) {
  return [2.5*s.B04, 2.5*s.B03, 2.5*s.B02, s.dataMask];
}
