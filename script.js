const { spawn } = require('child_process');

// Define the sentence to be manipulated
const sentence = 'The quick brown fox jumps over the lazy dog.';

// Spawn a new Python process
const pythonProcess = spawn('python', ['script.py']);

// Send the sentence to the Python process
pythonProcess.stdin.write(sentence);
pythonProcess.stdin.end();

// Listen for data from the Python process
pythonProcess.stdout.on('data', (data) => {
  const manipulatedSentence = data.toString();
  console.log(`Manipulated sentence: ${manipulatedSentence}`);
});

// Listen for errors from the Python process
pythonProcess.stderr.on('data', (data) => {
  console.error(`Error: ${data}`);
});

// Listen for the Python process to exit
pythonProcess.on('exit', (code) => {
  console.log(`Python process exited with code ${code}`);
});