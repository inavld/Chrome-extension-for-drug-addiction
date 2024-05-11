function classifyAndBlurParagraphs() {
  const paragraphs = Array.from(document.querySelectorAll('p'));
  const textData = paragraphs.map(p => p.innerText);

  chrome.runtime.sendMessage({ action: 'classifyParagraphs', url: window.location.href }, response => {
    if (!response.error) {
      const classifiedParagraphs = response.processed_paragraphs;

      paragraphs.forEach((p, index) => {
        if (classifiedParagraphs[index] && classifiedParagraphs[index].startsWith('<b>')) {
          p.style.filter = 'blur(5px)'; // If the paragraph gets classified as 'Addicted' then it's blurred
        }
      });
    } else {
      console.error(response.error);
    }
  });
}

classifyAndBlurParagraphs();
