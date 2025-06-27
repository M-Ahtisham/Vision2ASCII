var postURL = function(url) {
    var form = document.createElement("FORM");
    form.method = "POST";
    form.enctype = "application/x-www-form-urlencoded";
    form.style.display = "none";
    document.body.appendChild(form);
    form.action = url.replace(/\?(.*)/, function(_, urlArgs) {
        urlArgs.replace(/\+/g, " ").replace(/([^&=]+)=([^&=]*)/g, function(input, key, value) {
            input = document.createElement("INPUT");
            input.type = "hidden";
            input.name = decodeURIComponent(key);
            input.value = decodeURIComponent(value);
            form.appendChild(input);
        });
        return "";
    });
    form.submit();
    return false;
};

var changeColors = function(theme) {
    postURL(window.location.href + '?aktion=changetheme&theme=' + theme);
};

$(document).ready(function () {
    // toggle sidebar when button clicked
    $('.sidebar-toggle').on('click', function () {
        $('.sidebar').toggleClass('toggled');
    });

    // auto-expand submenu if an item is active
    var active = $('.sidebar .active');

    if (active.length && active.parent('.collapse').length) {
        var parent = active.parent('.collapse');

        parent.prev('a').attr('aria-expanded', true);
        parent.addClass('show');
    }
});