(() => {
  async function stats() {
    const feedback = document.getElementById('profile-stats-feedback');
    feedback.hidden = true;
    try {
      const { stats } = await Studio.request('/api/user_stats');
      for (const [id, key] of [
        ['totalSessions', 'total_sessions'],
        ['completedSessions', 'completed_sessions'],
        ['runningSessions', 'running_sessions'],
        ['totalFiles', 'total_files'],
      ])
        document.getElementById(id).textContent = stats[key] ?? '—';
      document.getElementById('storageUsed').textContent =
        stats.total_size_mb == null
          ? '—'
          : Number(stats.total_size_mb).toFixed(1) + ' MB';
    } catch (error) {
      feedback.hidden = false;
      Studio.state(feedback, 'error', error.message, stats);
    }
  }
  async function activity() {
    const root = document.getElementById('activityList');
    Studio.state(root, 'loading', '正在读取最近活动…');
    try {
      const data = await Studio.request('/api/user_activity');
      if (
        !data.activities.length ||
        data.activities.every((row) => row.status === 'info')
      )
        return Studio.state(
          root,
          'empty',
          '还没有训练记录，开始第一个实验吧。',
        );
      root.innerHTML = data.activities
        .map(
          (row) =>
            '<div class="activity-row"><span class="status ' +
            (row.status === 'completed'
              ? 'status-success'
              : row.status === 'failed'
                ? 'status-error'
                : 'status-running') +
            '">' +
            Studio.escape(row.status) +
            '</span><div><p>' +
            Studio.escape(row.text) +
            '</p><time>' +
            Studio.escape(row.time) +
            '</time></div></div>',
        )
        .join('');
    } catch (error) {
      Studio.state(root, 'error', error.message, activity);
    }
  }
  document.getElementById('profile-refresh').onclick = () => {
    stats();
    activity();
  };
  document.getElementById('profile-logout').onclick = async () => {
    try {
      await Studio.post('/api/logout');
      localStorage.removeItem(
        'rl4co_training_session:' + document.body.dataset.userId,
      );
      location.href = '/login';
    } catch (error) {
      Studio.toast(error.message, 'error');
    }
  };
  stats();
  activity();
})();
