//VERSION=3
function setup() {
  return { input: ["B08", "B12", "SCL", "dataMask"], output: { bands: 4 } };
}
function evaluatePixel(s) {
  const nbr = (s.B08 - s.B12) / (s.B08 + s.B12);
  const cloud = [3,8,9,10,11].includes(s.SCL);
  if (cloud || !s.dataMask) return [0,0,0,0];
  return [1-nbr, nbr, 0.1, 1];
}
