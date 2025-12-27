async function assign(folderRel, docId) {
  const res = await fetch(`/api/docs/${docId}/assign`, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({folder_rel_path: folderRel})
  });
  if (res.ok) {
    location.href = '/';
  } else {
    alert(await res.text());
  }
}

async function retry(docId) {
  const res = await fetch(`/api/docs/${docId}/retry`, {
    method: 'POST'
  });
  if (res.ok) {
    location.reload();
  } else {
    alert(await res.text());
  }
}
