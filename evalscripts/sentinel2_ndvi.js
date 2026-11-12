//VERSION=3
function setup() {
  return { input: ["B08", "B04", "SCL", "dataMask"], output: { bands: 4 } };
}
function evaluatePixel(s) {
  const ndvi = (s.B08 - s.B04) / (s.B08 + s.B04);
  const cloud = [3,8,9,10,11].includes(s.SCL);
  if (cloud || !s.dataMask) return [0,0,0,0];
  if (ndvi < 0) return [0.2, 0.2, 0.2, 1];
  return [Math.max(0, 1-ndvi), Math.min(1, ndvi), 0.15, 1];
}
