(() => {
  'use strict';
  const form = document.querySelector('[data-auth]'),
    button = document.getElementById('submitBtn'),
    banner = document.getElementById('alert');
  const error = (input, text) => {
    input.setAttribute('aria-invalid', String(Boolean(text)));
    const node = document.getElementById(input.id + '-error');
    node.textContent = text;
    node.hidden = !text;
    input.setAttribute('aria-describedby', node.id);
  };
  function confirmPassword() {
    const input = document.getElementById('confirm_password');
    if (!input) return true;
    const matches = input.value === document.getElementById('password').value;
    error(input, input.value && !matches ? '两次输入的密码不一致。' : '');
    return matches;
  }
  form.querySelectorAll('input').forEach((input) =>
    input.addEventListener('input', () => {
      error(input, '');
      confirmPassword();
    }),
  );
  if (form.dataset.auth === 'register')
    document.getElementById('password').addEventListener('input', (event) => {
      const value = event.target.value;
      document.getElementById('strengthText').textContent = value
        ? '密码强度：' +
          (value.length >= 10 && /[a-z]/i.test(value) && /\d/.test(value)
            ? '较强'
            : '建议混合使用字母、数字与符号')
        : '';
    });
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (button.disabled) return;
    banner.hidden = true;
    let first;
    for (const input of form.querySelectorAll('input')) {
      const valid = input.checkValidity();
      error(
        input,
        valid
          ? ''
          : input.validity.valueMissing
            ? '请填写此字段。'
            : input.type === 'email'
              ? '请输入有效的邮箱地址。'
              : '请检查输入长度。',
      );
      if (!valid && !first) first = input;
    }
    if (!confirmPassword()) {
      const input = document.getElementById('confirm_password');
      error(input, '两次输入的密码不一致。');
      first ||= input;
    }
    if (first) {
      first.focus();
      return;
    }
    Studio.busy(button, true);
    const original = button.textContent;
    button.textContent =
      form.dataset.auth === 'login' ? '正在登录…' : '正在创建…';
    try {
      const payload = {
        username: form.elements.username.value,
        password: form.elements.password.value,
      };
      if (form.dataset.auth === 'register')
        payload.email = form.elements.email.value;
      const response = await Studio.post('/api/' + form.dataset.auth, payload);
      banner.className = 'banner banner-success';
      banner.textContent = response.message;
      banner.hidden = false;
      location.href = form.dataset.auth === 'login' ? '/' : '/login';
    } catch (cause) {
      banner.className = 'banner banner-error';
      banner.textContent = cause.message;
      banner.hidden = false;
      Studio.busy(button, false);
      button.textContent = original;
    }
  });
})();
