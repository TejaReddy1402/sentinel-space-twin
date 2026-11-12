//VERSION=3
function setup() {
  return { input: ["B03", "B08", "SCL", "dataMask"], output: { bands: 4 } };
}
function evaluatePixel(s) {
  const ndwi = (s.B03 - s.B08) / (s.B03 + s.B08);
  const cloud = [3,8,9,10,11].includes(s.SCL);
  if (cloud || !s.dataMask) return [0,0,0,0];
  return ndwi > 0 ? [0.1, 0.45 + ndwi, 1.0, 1] : [0.08, 0.10, 0.14, 1];
}
