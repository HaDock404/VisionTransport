function getResult() {

    var form = document.getElementById('inputForm');
    var formData = new FormData(form);

    fetch('/predict', {
        method: 'POST',
        body: formData
      })
      .then(response => response.text())
      .then(data => {
        document.getElementById('result').innerHTML = data;
      })
      .catch(error => {
        console.error('Error:', error);
      });
}