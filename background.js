// this is a mediator file between the flask server and the active tab
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'classifyParagraphs') {  //passing the classification result to content.js for possible blurring
    fetch('http://127.0.0.1:5000/scrape_and_classify', { //requesting the classification output from the server's endpoint
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ url: message.url })
    })
    .then(response => response.json())
    .then(data => sendResponse(data))
    .catch(error => sendResponse({ error: error.message }));

    return true; 
  }
});
