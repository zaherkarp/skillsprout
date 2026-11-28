// profile-token.js
// Keeps each profile's owner token in this browser and adds it to API calls.
//
// Background (app/core/profile_ownership.py): when a profile is created, the
// server returns an owner token exactly once and stores only its hash. Every
// API call for that profile must send the token in the X-Profile-Token
// header, or the server answers 403. Without this, anyone could read or
// delete any profile by counting ids.
//
// Where the token lives: localStorage, one entry per profile id, so it
// survives reloads and browser restarts on this device. Anyone who can run
// script on this page could read it, as with any credential a browser holds,
// so the pages must keep untrusted text out of innerHTML.
//
// Moving a profile to another browser, or reclaiming one created before
// tokens existed: open /flow/<id>#token=<token>. The part after "#" never
// leaves the browser (it is not sent to the server or written to its logs);
// this script saves it and then removes it from the address bar.
(function () {
    "use strict";

    var HEADER = "X-Profile-Token";

    function key(userId) {
        return "skillsprout:profile-token:" + String(userId);
    }

    // localStorage can throw (private windows, blocked site data). A failed
    // read or write just means no saved token; the server then answers 403
    // and the page explains what to do.
    function save(userId, token) {
        try { window.localStorage.setItem(key(userId), token); } catch (e) { /* see above */ }
    }

    function get(userId) {
        try { return window.localStorage.getItem(key(userId)); } catch (e) { return null; }
    }

    // If the address ends in #token=..., save that token for userId and strip
    // it from the address bar and history, so it is not left on screen or in
    // the back button.
    function captureFromHash(userId) {
        var match = /^#token=([A-Za-z0-9_-]+)$/.exec(window.location.hash || "");
        if (!match) return false;
        save(userId, match[1]);
        try {
            window.history.replaceState(null, "", window.location.pathname + window.location.search);
        } catch (e) { /* the token is saved either way */ }
        return true;
    }

    // Headers for an API call about userId: the owner token (if this browser
    // has it) plus anything the caller passes, e.g. Content-Type.
    function headers(userId, extra) {
        var h = {};
        var token = get(userId);
        if (token) h[HEADER] = token;
        if (extra) {
            for (var name in extra) {
                if (Object.prototype.hasOwnProperty.call(extra, name)) h[name] = extra[name];
            }
        }
        return h;
    }

    window.SkillSproutToken = { save: save, get: get, captureFromHash: captureFromHash, headers: headers };
})();
