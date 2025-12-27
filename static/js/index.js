function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function toast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2000);
}

function getTagClass(status) {
  const map = {
    'READY': 'tag-ready',
    'PROCESSING': 'tag-processing',
    'ERROR': 'tag-error',
    'NEW': 'tag-new',
    'MOVED': 'tag-moved'
  };
  return map[status] || 'tag-new';
}

async function refreshQueue() {
  const res = await fetch('/api/queue');
  const data = await res.json();
  const el = document.getElementById('queue');
  const statsEl = document.getElementById('stats');

  statsEl.innerHTML = `
    <div class="stat"><span class="stat-dot ready"></span>${data.ready} Ready</div>
    <div class="stat"><span class="stat-dot processing"></span>${data.processing} Processing</div>
    <div class="stat"><span class="stat-dot error"></span>${data.error} Error</div>
    <div class="stat"><span class="stat-dot moved"></span>${data.moved} Moved</div>
  `;

  if (data.items.length === 0) {
    el.innerHTML = `
      <div class="empty-state" style="grid-column: 1 / -1;">
        <div class="empty-state-icon">📭</div>
        <div>No documents in queue</div>
        <div class="muted" style="margin-top: 8px;">Drop files into the input folder to get started</div>
      </div>
    `;
    return;
  }

  el.innerHTML = '';

  for (const d of data.items) {
    const card = document.createElement('div');
    card.className = 'card';
    card.draggable = true;
    card.dataset.docId = d.id;

    const sug = d.suggestions && d.suggestions.length ? d.suggestions[0] : null;
    const sugFolder = sug ? sug.folder_rel : '—';
    const sugScore = sug ? sug.score : '';

    card.innerHTML = `
      <div class="card-header">
        <div class="card-title" title="${escapeHtml(d.name)}">${escapeHtml(d.name)}</div>
        <div class="tag ${getTagClass(d.status)}">${escapeHtml(d.status)}</div>
      </div>
      <div class="suggestion-box">
        <div class="suggestion-label">Suggested folder</div>
        <div class="suggestion-value">${escapeHtml(sugFolder)} ${sugScore ? `<span class="suggestion-score">(${escapeHtml(String(sugScore))})</span>` : ''}</div>
      </div>
      <div class="card-actions">
        <a href="/doc/${encodeURIComponent(d.id)}">View Details</a>
        ${d.status === 'ERROR' ? `<button class="btn btn-sm" onclick="retryDoc(${parseInt(d.id)})">Retry</button>` : ''}
      </div>
    `;

    card.addEventListener('dragstart', (e) => {
      e.dataTransfer.setData('text/plain', String(d.id));
      card.style.opacity = '0.5';
    });
    card.addEventListener('dragend', () => {
      card.style.opacity = '1';
    });

    el.appendChild(card);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('[data-folder-rel]').forEach(node => {
    node.addEventListener('dragover', (e) => {
      e.preventDefault();
      node.classList.add('drop-hover');
    });
    node.addEventListener('dragleave', () => node.classList.remove('drop-hover'));
    node.addEventListener('drop', async (e) => {
      e.preventDefault();
      node.classList.remove('drop-hover');

      const docId = e.dataTransfer.getData('text/plain');
      const folderRel = node.dataset.folderRel;

      const res = await fetch(`/api/docs/${docId}/assign`, {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({folder_rel_path: folderRel})
      });

      if (res.ok) {
        toast(`✓ Moved to ${folderRel}`);
        refreshQueue();
      } else {
        const msg = await res.text();
        alert(msg);
      }
    });
  });

  refreshQueue();
  setInterval(refreshQueue, 5000);
});

async function retryDoc(docId) {
  const res = await fetch(`/api/docs/${docId}/retry`, { method: 'POST' });
  if (res.ok) {
    toast('Retry queued');
    refreshQueue();
  } else {
    alert(await res.text());
  }
}
