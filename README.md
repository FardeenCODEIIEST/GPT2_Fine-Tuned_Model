## Backend For the GPT2-Fine-Tuned Model

### Requirements

  <ol>
    <span style="font-weight:bold">Python Library Requirements</span>
    <br></br>
    <li>Flask</li>
    <li>flask-cors</li>
    <li>torch</li>
    <li>transformers</li>
    <li>waitress</li>
  </ol>
  <ol>
    <span style="font-weight:bold">Optional System Requirements</span>
    <br></br>
    <li>CUDA-support is preferred </li>

  </ol>

### Run the App

To run the app just clone this repository first and then move into the repository directory, then open a terminal there and type the following command <code style="font-weight:bold">python ./server.py </code>

### Inference

Server will start-up at port 9000, after installing necessary model safe-tensors and configuration files.

To test the server, you use `postman` and do a `POST` request at `http://localhost:9000/` of the form(JSON)

```
{
  "prompt": "<Your starting sentence words>",
  "type":   "<short/medium/long>"
}
```

After sometime, you will get a response of the form(JSON)

```
{
  "generated_text": <Generated Response>
}
```
