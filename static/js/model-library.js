(() => {
  const search = document.getElementById('model-search'),
    category = document.getElementById('model-category'),
    support = document.getElementById('model-support'),
    cards = [...document.querySelectorAll('[data-model-card]')];
  function filter() {
    let count = 0;
    for (const card of cards) {
      const match =
        card.dataset.name
          .toLowerCase()
          .includes(search.value.trim().toLowerCase()) &&
        (!category.value || card.dataset.category === category.value) &&
        (!support.value || card.dataset.integrated === 'true');
      card.hidden = !match;
      if (match) count++;
    }
    document.getElementById('model-empty').hidden = count !== 0;
    document.getElementById('model-filter-count').textContent =
      count + ' / ' + cards.length + ' 个模型与方法';
  }
  search.oninput = filter;
  category.onchange = filter;
  support.onchange = filter;
})();
