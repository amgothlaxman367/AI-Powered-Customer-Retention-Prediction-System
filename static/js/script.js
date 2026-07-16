/*=========================================================
 AI Powered Customer Retention Prediction System
 Premium JavaScript
=========================================================*/

document.addEventListener("DOMContentLoaded", function () {

    /* ==========================
       Popup
    ========================== */

    const popup = document.getElementById("predictionModal");

    if (popup) {

        popup.style.display = "flex";

    }

    window.closePopup = function () {

        if (popup) {

            popup.style.display = "none";

        }

    };

    const closeBtn = document.querySelector(".close-popup");

    if (closeBtn) {

        closeBtn.addEventListener("click", function () {

            popup.style.display = "none";

        });

    }

    window.addEventListener("click", function (e) {

        if (e.target === popup) {

            popup.style.display = "none";

        }

    });


    /* ==========================
       Form Validation
    ========================== */

    const form = document.getElementById("predictionForm");

    if (form) {

        form.addEventListener("submit", function (e) {

            const selects = document.querySelectorAll("select");

            let valid = true;

            selects.forEach(function (item) {

                if (item.value === "") {

                    valid = false;

                    item.style.border = "2px solid red";

                }

                else {

                    item.style.border = "1px solid rgba(255,255,255,.20)";

                }

            });

            if (!valid) {

                e.preventDefault();

                alert("Please select all required fields.");

                return;

            }

            const button = document.querySelector(".predict-btn");

            button.disabled = true;

            button.innerHTML =

            '<i class="fa-solid fa-spinner fa-spin"></i> Predicting...';

        });

    }


    /* ==========================
       Hover Animation
    ========================== */

    const cards = document.querySelectorAll(

        ".glass-card,.stat-card,.ai-card,.profile-card,.logo-card"

    );

    cards.forEach(function(card){

        card.addEventListener("mouseenter",function(){

            this.style.transform="translateY(-8px)";

        });

        card.addEventListener("mouseleave",function(){

            this.style.transform="translateY(0px)";

        });

    });


    /* ==========================
       Fade Animation
    ========================== */

    const observer = new IntersectionObserver(function(entries){

        entries.forEach(function(entry){

            if(entry.isIntersecting){

                entry.target.style.opacity="1";

                entry.target.style.transform="translateY(0px)";

            }

        });

    },{

        threshold:0.15

    });

    document.querySelectorAll(

        ".stat-card,.glass-card,.ai-card,.footer"

    ).forEach(function(el){

        el.style.opacity="0";

        el.style.transform="translateY(40px)";

        el.style.transition=".8s";

        observer.observe(el);

    });


    /* ==========================
       Mouse Glow Effect
    ========================== */

    document.addEventListener("mousemove", function(e){

        const glow = document.querySelector(".bg-animation");

        if(glow){

            glow.style.backgroundPosition =

            `${e.clientX/40}px ${e.clientY/40}px`;

        }

    });


    /* ==========================
       Smooth Button Effect
    ========================== */

    const btn=document.querySelector(".predict-btn");

    if(btn){

        btn.addEventListener("mouseenter",function(){

            this.style.boxShadow=

            "0 0 20px cyan,0 0 50px #00e5ff";

        });

        btn.addEventListener("mouseleave",function(){

            this.style.boxShadow=

            "0 0 15px rgba(0,255,255,.35)";

        });

    }

});